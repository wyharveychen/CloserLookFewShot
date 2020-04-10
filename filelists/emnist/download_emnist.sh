#!/usr/bin/env bash
wget https://github.com/NanqingD/DAOSL/raw/master/data/emnist.zip 
unzip emnist.zip
python3 invert_emnist.py
python3 write_cross_char_valnovel_filelist.py
