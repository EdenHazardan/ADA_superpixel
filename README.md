# ADA_superpixel
This repository is released for double-blind submission, which can reproduce the main results (our proposed superpixel-level method for active domain adaptation in semantic segmentation, ADA_superpixel) of the experiment on GTA5 to Cityscapes. Experiments on the SYNTHIA to Cityscapes can be easily implemented by slightly modifying the dataset and setting. Notably, we use DACS as UDA-merge here for simplify, while the implementation of daformer uses the mmsegmentation framework.

## Install & Requirements

The code has been tested on pytorch=1.8.0 and python3.8. Please refer to ``requirements.txt`` for detailed information.

### To Install python packages
```
pip install -r requirements.txt
```

## Download Pretrained Weights
For the segmentation model initialization, following DDM, we start with a model pretrained on ImageNet: [Download](http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth)


## Data preparation
You need to download the [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) datasets and [Cityscapes](https://www.cityscapes-dataset.com/) datasets.

Your directory tree should be look like this:
```
./DSTC-SSDA/data
├── cityscapes
|  ├── gtFine
|  |  |—— train
|  |  └── val
|  └── leftImg8bit
│       ├── train
│       └── val
├── GTA5
|  ├── images
|  └── labels 
```

## Superpixel Generation
We use SSN to generate superpixels for Cityscapes data and follow [SSN](https://github.com/perrying/ssn-pytorch) to train SSN on source domain (GTA5 or SYNTHIA). Then we save the superpixel results at '/home/gaoy/ssn-pytorch-patch-1/SSN_city'.

## Labeling Phase 1 

```
# get active label Y1
bash ADA_superpixel/exp/Active_label/Labeling_phase_1/script/train.sh
```

## Training Target-base

```
# use active label Y1 to train Target-base
bash ADA_superpixel/exp/Training/Target_base/script/train.sh
```

## Labeling Phase 2

```
# get active label Y2
bash ADA_superpixel/exp/Active_label/Labeling_phase_2/script/train.sh
```

## Training Final model

```
# use active label Y2 to train Final model
bash ADA_superpixel/exp/Training/Target_base/script/train.sh
```











