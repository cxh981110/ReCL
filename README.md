# ReCL
**Authors**: Xuhui Chang,Junhai Zhai, Shaoxin Qiu

## Installation

**Requirements**

* Python 3.7
* torchvision 0.4.0
* Pytorch 1.6.0

**Dataset Preparation**
* [CIFAR-10 & CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
* [ImageNet](http://image-net.org/index)

Change the `data_path` in `config/*/*.yaml` accordingly.

## Training:

To train a model:

(one GPU for CIFAR-10-LT & CIFAR-100-LT, four GPUs for ImageNet-LT)

```
python train_stage.py --cfg ./config/DATASETNAME/DATASETNAME_ARCH_stage1_mixup.yaml
```

`DATASETNAME` can be selected from `cifar10`,  `cifar100`, `imagenet`, `ina2018`, and `places`.

`ARCH` can be `resnet32` for `cifar10/100`, `resnet50/101/152` for `imagenet`.



The saved folder (including logs and checkpoints) is organized as follows.
```
ReCL
├── saved
│   ├── modelname_date
│   │   ├── ckps
│   │   │   ├── current.pth.tar
│   │   │   └── model_best.pth.tar
│   │   └── logs
│   │       └── modelname.txt
│   ...   
```
## 
