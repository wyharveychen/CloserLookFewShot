#!/usr/bin/env bash
wget https://raw.githubusercontent.com/twitter/meta-learning-lstm/master/data/miniImagenet/train.csv 
wget https://raw.githubusercontent.com/twitter/meta-learning-lstm/master/data/miniImagenet/val.csv 
wget https://raw.githubusercontent.com/twitter/meta-learning-lstm/master/data/miniImagenet/test.csv
wget http://image-net.org/image/ILSVRC2015/ILSVRC2015_CLS-LOC.tar.gz 
tar -zxvf ILSVRC2015_CLS-LOC.tar.gz
python3 write_miniImagenet_filelist.py
python3 write_cross_filelist.py
