

## About the project
This repository includes the codes of two papers:
1. [Complexity-aware Adaptive Training and Inference for Edge-Cloud Distributed AI Systems](https://ieeexplore.ieee.org/abstract/document/9546405)
2. [Conditionally Deep Hybrid Neural Networks Across Edge and Cloud](https://arxiv.org/abs/2005.10851)
Our project is built on the framework of Distiller. Please refer to the website (https://github.com/NervanaSystems/distiller.git) for instructions for installation and environment settings.
Clone the project repository from github:
```
$ git clone https://github.com/yinghanlong/Complexity-aware-AI.git
```

## Getting Started
 After setting up the enviroment, you can run experiments following the example commands in ```run.sh```. For example, to train an adaptive neural network based on the pretrained ImageNet, you can use the following.

```
$ source env/bin/activate
$ cd examples/classifier_compression/
$ python Block-Train-ImageNet-pretrain.py --arch=resnet18_p --epoch=110 -b 256 --lr=0.01 -j 1 --out-dir . -n imagenet . --earlyexit_lossweights 0.3 --earlyexit_thresholds 0.8 --deterministic --gpus=1
```
To train MobileNetV2,
```
$python Mobilenet-extend.py --arch=mobilenet_v2 --epoch=180 -b 128 --lr=0.01 -j 1 --out-dir . -n mobilenet . --deterministic --gpus=2 --earlyexit_lossweights 0.3 --earlyexit_thresholds 0.8
```

Use ```--evaluate``` and ```--resume=model_dir``` to load a trained model and run evaluation.

For more details, there are files you can refer to:
+ Models for CIFAR10/100 (modified ResNets into early exiting models and MEANet models which includes main, adaptive and extension blocks): ```/models/cifar10```
+ Models for ImageNet (ResNets and MobileNetV2):```/models/imagenet```
+ Codes for training MEANet models which includes main, adaptive and extension blocks: ```/examples/classifier_compression/Block-Train-extend.py```,```/examples/classifier_compression/Mobilenet-extend.py```,```/examples/classifier_compression/Block-Train-ImageNet-pretrained.py```
+ Codes for training hybrid quantized models with early exits:```/examples/classifier_compression/early-exit-classifier.py```
+ Setting K-bit quantization or binarization for specific layers:```/examples/classifier_compression/util_bin.py```
+ Examples of hard classes of ImageNet/CIFAR100:  ```examples/classifier_compression/mobilenet_imagenet/hard_classes.pickle```. ```examples/classifier_compression/resnet32_hardclass/hard_classes.pickle```

We will add more explantions and commands later. Please email me long273@purdue.edu if you have any questions regarding the project or the codes.


## Built With

* [PyTorch](http://pytorch.org/) - The tensor and neural network framework used by Distiller.



## Citation

If you used for your work, please use the following citation:

```
@INPROCEEDINGS{9546405,

  author={Long, Yinghan and Chakraborty, Indranil and Srinivasan, Gopalakrishnan and Roy, Kaushik},

  booktitle={2021 IEEE 41st International Conference on Distributed Computing Systems (ICDCS)}, 

  title={Complexity-aware Adaptive Training and Inference for Edge-Cloud Distributed AI Systems}, 

  year={2021},

  volume={},

  number={},

  pages={573-583},

  doi={10.1109/ICDCS51616.2021.00061}}

@misc{https://doi.org/10.48550/arxiv.2005.10851,
  doi = {10.48550/ARXIV.2005.10851},
  
  url = {https://arxiv.org/abs/2005.10851},
  
  author = {Long, Yinghan and Chakraborty, Indranil and Roy, Kaushik},
  
  
  
  title = {Conditionally Deep Hybrid Neural Networks Across Edge and Cloud},
  
  publisher = {arXiv},
  
  year = {2020},
  
}

```
