# A Closer Look at Few-shot Classification
Source code to ICLR'19, 'A Closer Look at Few-shot Classification' (still under construction)

This is a PyTorch implementation of our paper [A Closer Look at Few-shot Classification](https://openreview.net/pdf?id=HkxLXnAcFQ) accepted by ICLR 2019.

A detailed empirical study in few-shot classification with an integrated testbed

## Enviroment
 - Python3
 - [Pytorch](http://pytorch.org/)
 - json

## Setting preparation
#CUB
* Change directory to `./filelists/CUB`
* run `source ./download_CUB.sh`

#mini-ImageNet
* Change directory to `./filelists/miniImagenet`
* run `source ./download_miniImagenet.sh` 

(WARNING: This would download the 155G ImageNet dataset. You can comment out correponded line 5-6 in `download_miniImagenet.sh` if you already have one.) 

#mini-ImageNet->CUB
* Finish preparation for CUB and mini-ImageNet and you are done!

#omniglot (extra experiment in camera-ready version)
* Change directory to `./filelists/omniglot`
* run `source ./download_omniglot.sh` 

#omniglot->emnist (extra experiment in camera-ready version)
* Finish preparation for omniglot first
* Change directory to `./filelists/emnist`
* run `source ./download_emnist.sh`  

#self-defined setting
* Require 3 data split json file: 'base.json', 'val.json', 'novel.json' for each dataset  
* The format should look like  
{"label_names": ["class0","class1",...], "image_names": ["filepath1","filepath2",...],"image_labels":[l1,l2,l3,...]}  
See test.json for reference
* Put these file in the same folder and change data_dir['DATASETNAME'] in configs.py to the folder path  

## Train
Run
```python ./train.py --dataset [DATASETNAME] --model [BACKBONENAME] --method [METHODNAME] [--OPTIONARG]```

For example, run `python ./train.py --dataset miniImagenet --model Conv4 --method baseline --train_aug`  
Commands below follow this example, and please refer to io_utils.py for more options

## Save features
Save feature before classifaction layer to increase test speed, not applicable to MAML, but required for other methods  
Run
```python ./save_features.py --dataset miniImagenet --model Conv4 --method baseline --train_aug```

## Test
Run
```python ./test.py --dataset miniImagenet --model Conv4 --method baseline --train_aug```

## Results
* The test results would be recorded in `./record/results.txt`
* Also, we have put our source file of figure in the `./record/few_shot_exp_figures.xlsx` in case you want to add your result to it.

## Reference
This testbed has modified and integrated the following codes:

* Framework, Backbone, Method: Matching Network
https://github.com/facebookresearch/low-shot-shrink-hallucinate 
* Method: Prototypical Network
https://github.com/jakesnell/prototypical-networks
* Method: Relational Network
https://github.com/floodsung/LearningToCompare_FSL
* Method: MAML
https://github.com/cbfinn/maml  
https://github.com/dragen1860/MAML-Pytorch  
https://github.com/katerakelly/pytorch-maml

## Citation
Please use the bibtex:
```
@inproceedings{
chen2019closerfewshot,
title={A Closer Look at Few-shot Classification},
author={Wei-Yu Chen and Yen-Cheng Liu and Zsolt Kira and Yu-Chiang Frank Wang and Jia-Bin Huang},
booktitle={International Conference on Learning Representations},
year={2019}
}
```
