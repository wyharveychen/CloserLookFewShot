#!/usr/bin/env bash
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xvzf cifar-100-python.tar.gz
python3 write_CIFARFS_filelist.py
