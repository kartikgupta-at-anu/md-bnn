# Mirror Descent View for Neural Network Quantization

This repository is the official implementation of AISTATS 2021 paper: [Mirror Descent View for Neural Network Quantization](https://arxiv.org/pdf/1910.08237.pdf).

This code is for research purposes only.

Any questions or discussions are welcomed!

## Installation and Setup

Setup python virtual environment.

```
virtualenv -p python2.7 venv
source venv/bin/activate
pip install -r requirements.txt
```

NOTE: To produce the exact same results, please use NVIDIA Tesla P100 GPUs and exact same versions of libraries as mentioned below. Otherwise due to numerical issues and randomness results might differ slightly from the paper.

The code has been tested with `CUDA 9.1.85 `, `PyTorch 0.4.0`, `opencv 3.2.0` and `torchvision 0.2.1` in `Python 2.7.13`.

TINYIMAGENET dataset can be downloaded from [here](https://drive.google.com/file/d/1nkFCYsEpT5lj7sXaAXhlAaD1F4GTJSru/view?usp=sharing).

## Training with MD for NN Quantization

Shell scripts to train different MD versions proposed in the paper can be found in `train_scripts/` folder. 

```
sh train_scripts/cifar10resnet18.sh
sh train_scripts/cifar100resnet18.sh
sh train_scripts/cifar10vgg16.sh
sh train_scripts/cifar100vgg16.sh
sh train_scripts/tinyimagenet_resnet18.sh
```

## Evaluation 

Download the saved models trained using MD on different datasets and architectures from [here](https://drive.google.com/file/d/1E93Vpe-CRQpbEqZwC9UexY5uSpwlFjcA/view?usp=sharing).

Shell scripts to evaluate models trained using different MD versions proposed in the paper can be found in `test_scripts/` folder. 

```
sh test_scripts/cifar10resnet18.sh
sh test_scripts/cifar100resnet18.sh
sh test_scripts/cifar10vgg16.sh
sh test_scripts/cifar100vgg16.sh
sh test_scripts/tinyimagenet_resnet18.sh
```

## Cite

If you make use of this code in your own work, please cite our papers:

```
@inproceedings{ajanthan2019mirror,
  title={Mirror descent view for neural network quantization},
  author={Ajanthan, Thalaiyasingam and Gupta, Kartik and Torr, Philip HS and Hartley, Richard and Dokania, Puneet K},
  booktitle={Artificial intelligence and statistics},
  year={2021},
  organization={PMLR}
}
@inproceedings{ajanthan2019proximal,
  title={Proximal mean-field for neural network quantization},
  author={Ajanthan, Thalaiyasingam and Dokania, Puneet K and Hartley, Richard and Torr, Philip HS},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4871--4880},
  year={2019}
}
```

#### Contact
Kartik Gupta (kartik.gupta@anu.edu.au).

Acknowledgements
----------------------
* [PMF](https://github.com/tajanthan/pmf), for PMF Code
* [BNN](https://github.com/itayhubara/BinaryNet.pytorch), for some utility functions.
* [Models Defs](https://github.com/kuangliu/pytorch-cifar), for model definitions.
