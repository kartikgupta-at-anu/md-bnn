"""MD-SOFTMAX method for NN Quantization
"""

import logging
import os
from timeit import default_timer as timer

import optimizers as coptim
import utils.simplex_utils as su
import utils.utils as util

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cfgs.cfg as cfg_exp

import models

BETAMAX = 10.0**35
BEST_ACC = 0.0


class auxmodel():
    """ storing auxiliary parameters
    """
    def __init__(self, model):
        self.nparams = 0
        self.auxparams = []
        for i, p in enumerate(model.parameters()):
            self.auxparams.append(p.data.clone())
            self.nparams += 1
        print 'No of param-sets: {0}'.format(self.nparams)
        return

    def store(self, model):
        for i, p in enumerate(model.parameters()):
            self.auxparams[i].copy_(p.data)

    def restore(self, model):   # updates model.parameters!
        for i, p in enumerate(model.parameters()):
            p.data.copy_(self.auxparams[i])


def doround(model, device, scheme='ARGMAX', data=None, target=None, optimizer=None, criterion=None):
    """ do rounding given the feasible point in the polytope
    """
    if scheme == 'ARGMAX':
        for i, p in enumerate(model.parameters()):
            w = su.view_w_as_u(p.data, model.qlevels) # convert to desired matrix format (N x d)

            wi = w.argmax(dim=1, keepdim=True)
            ids = torch.arange(model.qlevels, dtype=torch.long, device=device)
            ids = ids.repeat(w.size(0), 1)
            w = ids.eq(wi)
            w = w.float()
            p.data = su.view_u_as_w(w, p.data)   # convert N x d to original format
    elif scheme == 'ICM':
        if data is None or target is None or optimizer is None or criterion is None:
            pass    # use last minibatch gradient
        else:
            pass
        for i, p in enumerate(model.parameters()):
            p.data = su.icm(model.qlevels, p.grad.data)    
        
    else:
        print 'Rounding type "{0}" not recognized, returning ...'.format(scheme)
        return


def simplex(model, device, projection='SOFTMAX', beta=1):
    """ project the parameters to the feasible polytope
    """
    if projection == 'ARGMAX':  # argmax based projection: rounding!
        doround(model, device, scheme='ARGMAX')
        return

    for i, p in enumerate(model.parameters()):
        w = su.view_w_as_u(p.data, model.qlevels) # convert to desired matrix format (N x d)

        if projection == 'SOFTMAX': # softmax based simplex projection
            w = F.softmax(w * beta, dim=1)
        elif projection == 'EUCLIDEAN':   # condat based (Euclidean) simplex projection 
            w = su.sparsemax(w, model.qlevels)
        elif projection == 'ICM': # icm based simplex projection
            pass
        else:
            print 'Projection type "{0}" not implemented, returning ...'.format(projection)
            
        p.data = su.view_u_as_w(w, p.data)   # convert N x d to original format
        assert(su.isfeasible(model.qlevels, p.data))


def train_step(model, device, data, target, optimizer, criterion, take_step=True):
    """ training step given the mini-batch
        depending on take_step, this is used for gradient computation
    """
    data, target = data.to(device, torch.float), target.to(device, torch.long)
    data.requires_grad_(True)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    if take_step:   # hack for use this code to compute gradients
        optimizer.step()
    return loss.item()


def evaluate(args, model, device, loader, training=False, amodel=None, beta=1., summary_writer=None, iterations=None):
    """ evaluate the model given data
    """
    global BEST_ACC
    model.eval()
    correct1 = 0
    correct5 = 0
    tsize = 0

    if training:
        assert(amodel is not None)
        # store aux-weights
        amodel.store(model)
        # projection and rounding
        doround(model, device, scheme=args.rounding)
        if summary_writer is not None:
            for name, param in model.named_parameters():
                summary_writer.add_histogram(name, param.clone().cpu().data.numpy(), iterations)
            plus_ones = 0
            minus_ones = 0
            for name, param in model.named_parameters():
                plus_ones += torch.sum(param==1)
                minus_ones += torch.sum(param==-1)
            summary_writer.add_scalar('plus_ones', int(plus_ones), iterations)
            summary_writer.add_scalar('minus_ones', int(minus_ones), iterations)

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device, torch.float), target.to(device, torch.long)
            output = model(data)
            # topk accuracy
            c1, c5 = util.accuracy(output.data, target, topk=(1,5))
            correct1 += c1
            correct5 += c5
            tsize += target.size(0)

    if training:
        # restore aux-weights
        amodel.restore(model)
        if summary_writer is not None:
            for name, param in model.named_parameters():
                summary_writer.add_histogram(name + '_unquantized', param.clone().cpu().data.numpy(), iterations, bins=1000)

        model.train()

    acc1 = 100. * correct1 / tsize
    acc5 = 100. * correct5 / tsize
    if (acc1 > BEST_ACC):
        BEST_ACC = acc1.item()
        if training:    # storing the continuous weights of the best model, done separately from checkpoint!
            util.save_model({'state_dict': model.state_dict(), 'best_acc1': BEST_ACC, 'beta': beta}, args.save_name) 
    
    return acc1.item(), acc5.item()


def init_weights(model, Q_l, device, xavier=False):
    for p in model.parameters():
        if xavier and len(p.size()) >= 2:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.normal_(p, std=0.1)
    simplex(model, device, beta=1.0)


def set_weights(device, params, w1, w2):
    for i, p in enumerate(params):
        if i == 0:
            p.data.copy_(w1)
        elif i == 1:
            p.data.copy_(w2)
        else:
            assert 0


def setup_and_run(args, criterion, device, train_loader, test_loader, val_loader, logging, results, summary_writer):
    global BEST_ACC
    print('\n#### Running mda-net ####')

    # quantized levels
    if args.quant_levels <= 1:
        print 'Quantization levels "{0}" is invalid, exiting ...'.format(args.quant_levels)
        exit()
    qs = args.quant_levels/2
    qe = args.quant_levels/2+1
    Q_l = torch.arange(-qs, qe)    
    if args.quant_levels % 2 == 0:  # if even remove 0
        Q_l = torch.cat((Q_l[:qs], Q_l[qe:]))
    Q_l = Q_l.to(device, dtype=torch.float)

    beta = 1.0    # discrete forcing scalar, used only for softmax based projection

    # architecture
    if 'VGG' in args.architecture:
        assert(args.architecture == 'VGG11' or args.architecture == 'VGG13' or args.architecture == 'VGG16' 
                or args.architecture == 'VGG19')
        model = models.SVGG(args.architecture, Q_l, args.input_channels, args.im_size, args.output_dim).to(device)
    elif args.architecture == 'RESNET18':
        model = models.SResNet18(Q_l, args.input_channels, args.im_size, args.output_dim).to(device)
    else:
        print 'Architecture type "{0}" not recognized, exiting ...'.format(args.architecture)
        exit()

    # optimizer
    if args.optimizer == 'MDA_SOFTMAX_ADAM':
        optimizer = coptim.EGDadam(model.parameters(), model.qlevels, lr=args.learning_rate, weight_decay=args.weight_decay
                                   , beta_mda=beta)
    else:
        print 'Optimizer type "{0}" not recognized, exiting ...'.format(args.optimizer)
        exit()
    
    # lr-scheduler
    if args.lr_decay == 'STEP':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_scale)
    elif args.lr_decay == 'EXP':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_scale)
    elif args.lr_decay == 'MSTEP':
        x = args.lr_interval.split(',')
        lri = [int(v) for v in x]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lri, gamma=args.lr_scale)
        args.lr_interval = 1    # lr_interval handled in scheduler!
    else:
        print 'LR decay type "{0}" not recognized, exiting ...'.format(args.lr_decay)
        exit()

    init_weights(model, Q_l, device, xavier=True)
    if not args.eval:
        logging.info(model)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    if not args.eval:
        logging.info("Number of parameters: %d", num_parameters)

    start_epoch = -1
    iters = 0   # total no of iterations, used to do many things!
    amodel = auxmodel(model)
    # optionally resume from a checkpoint
    if args.eval:
        logging.info('Loading checkpoint file "{0}" for evaluation'.format(args.eval))
        if not os.path.isfile(args.eval):
            print 'Checkpoint file "{0}" for evaluation not recognized, exiting ...'.format(args.eval)
            exit()
        checkpoint = torch.load(args.eval)
        model.load_state_dict(checkpoint['state_dict'])
        beta = checkpoint['beta']
        logging.debug('beta: {0}'.format(beta))

    elif args.resume:
        checkpoint_file = args.resume
        logging.info('Loading checkpoint file "{0}" to resume'.format(args.resume))
        if not os.path.isfile(checkpoint_file):
            print 'Checkpoint file "{0}" not recognized, exiting ...'.format(checkpoint_file)
            exit()
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        assert(args.architecture == checkpoint['architecture'])
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        BEST_ACC = checkpoint['best_acc1']
        beta = checkpoint['beta']
        iters = checkpoint['iters']
        logging.debug('best_acc1: {0}, beta: {1}, iters: {2}'.format(BEST_ACC, beta, iters))

    batch_per_epoch = len(train_loader)

    if not args.eval:
        logging.info('Training...')
        model.train()
        st = timer()                

        for e in range(start_epoch + 1, args.num_epochs):
            for i, (data, target) in enumerate(train_loader):
                l = train_step(model, device, data, target, optimizer, criterion)
                if i % args.log_interval == 0:
                    acc1, acc5 = evaluate(args, model, device, val_loader, training=True, amodel=amodel,
                                          beta=beta, summary_writer=summary_writer, iterations=e*batch_per_epoch+i)
                    logging.info('Epoch: {0},\t Iter: {1},\t Loss: {loss:.5f},\t Val-Acc1: {acc1:.2f} '
                                 '(Best: {best:.2f}),\t Val-Acc5: {acc5:.2f}'.format(e, i, 
                                     loss=l, acc1=acc1, best=BEST_ACC, acc5=acc5))
    
                if iters % args.beta_interval == 0:
                    beta = min(beta * args.beta_scale, BETAMAX)
                    optimizer.beta_mda = beta
                    logging.info('beta: {0}'.format(beta))

                if iters % args.lr_interval == 0:
                    lr = args.learning_rate
                    for param_group in optimizer.param_groups:
                        lr = param_group['lr']                        
                    scheduler.step()
                    for param_group in optimizer.param_groups:
                        if lr != param_group['lr']:
                            logging.info('lr: {0}'.format(param_group['lr']))   # print if changed
                iters += 1

            # save checkpoint
            acc1, acc5 = evaluate(args, model, device, val_loader, training=True, amodel=amodel, beta=beta)
            results.add(epoch=e, iteration=i, train_loss=l, val_acc1=acc1, best_val_acc1=BEST_ACC)
            util.save_checkpoint({'epoch': e, 'architecture': args.architecture, 'state_dict': model.state_dict(), 
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 
                'best_acc1': BEST_ACC, 'iters': iters, 'beta': beta}, is_best=False, path=args.save_dir)
            results.save()
    
        et = timer()
        logging.info('Elapsed time: {0} seconds'.format(et - st))
    
        acc1, acc5 = evaluate(args, model, device, val_loader, training=True, amodel=amodel, beta=beta)
        logging.info('End of training, Val-Acc: {acc1:.2f} (Best: {best:.2f}), Val-Acc5: {acc5:.2f}'.format(acc1=acc1, 
            best=BEST_ACC, acc5=acc5))
        # load saved model
        saved_model = torch.load(args.save_name)   
        model.load_state_dict(saved_model['state_dict'])
        beta = saved_model['beta']
    # end of training
    
    # eval-set
    if args.eval_set != 'TRAIN' and args.eval_set != 'TEST':
        print 'Evaluation set "{0}" not recognized ...'.format(args.eval_set)

    logging.info('Evaluating fractional mda-net on the {0} set...'.format(args.eval_set))
    st = timer()                
    if args.eval_set == 'TRAIN':
        acc1, acc5 = evaluate(args, model, device, train_loader)
    else: 
        acc1, acc5 = evaluate(args, model, device, test_loader)
    et = timer()
    logging.info('Accuracy: top-1: {acc1:.2f}, top-5: {acc5:.2f}%'.format(acc1=acc1, acc5=acc5))
    logging.info('Elapsed time: {0} seconds'.format(et - st))

    doround(model, device, scheme=args.rounding)    # icm rounding not supported!
    logging.info('Evaluating discrete mda-net on the {0} set...'.format(args.eval_set))
    st = timer()                
    if args.eval_set == 'TRAIN':
        acc1, acc5 = evaluate(args, model, device, train_loader)
    else: 
        acc1, acc5 = evaluate(args, model, device, test_loader)
    et = timer()
    logging.info('Accuracy: top-1: {acc1:.2f}, top-5: {acc5:.2f}%'.format(acc1=acc1, acc5=acc5))
    logging.info('Elapsed time: {0} seconds'.format(et - st))
