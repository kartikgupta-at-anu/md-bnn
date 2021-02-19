#!/usr/bin/env bash

dt='CIFAR100'
sd=./out
ep=0 #0:use iters
bs=128
ar='RESNET18'
lf='CROSSENTROPY'
gpuid=0

#### MD-Softmax
op='MDA_SOFTMAX_ADAM'
pr='SOFTMAX'
mt='MDA_SOFTMAX'
evalfile='saved_models/CIFAR100/RESNET18/MDA_SOFTMAX/best_model.pth.tar'

python quantized_nets.py --gpu-id $gpuid --save-dir $sd --architecture $ar --loss-function $lf --method $mt  --projection $pr --dataset $dt --batch-size $bs --eval $evalfile --op $op

#### MD-Softmax-S
op='ADAM'
pr='SOFTMAX'
mt='SOFTMAX_PROJECTION'
evalfile='saved_models/CIFAR100/RESNET18/SOFTMAX_PROJECTION/best_model.pth.tar'

python quantized_nets.py --gpu-id $gpuid --save-dir $sd --architecture $ar --loss-function $lf --method $mt --projection $pr --dataset $dt --batch-size $bs --full-ste --eval $evalfile --op $op

#### MD-tanh
op='MDA_TANH_ADAM'
pr='TANH'
mt='MDA_TANH'
evalfile='saved_models/CIFAR100/RESNET18/MDA_TANH/best_model.pth.tar'

python quantized_nets.py --gpu-id $gpuid --save-dir $sd --architecture $ar --loss-function $lf --method $mt --projection $pr --dataset $dt --batch-size $bs --tanh --eval $evalfile --op $op

#### MD-tanh-S
op='ADAM'
pr='TANH'
mt='TANH_PROJECTION'
evalfile='saved_models/CIFAR100/RESNET18/TANH_PROJECTION/best_model.pth.tar'

python quantized_nets.py --gpu-id $gpuid --save-dir $sd --architecture $ar --loss-function $lf --method $mt --projection $pr --dataset $dt --batch-size $bs --tanh --full-ste --eval $evalfile --op $op