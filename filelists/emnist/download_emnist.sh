#!/usr/bin/env bash
wget https://github.com/NanqingD/DAOSL/raw/master/data/emnist.zip 
unzip emnist.zip
python invert_emnist.py
python write_cross_char_valnovel_filelist.py
