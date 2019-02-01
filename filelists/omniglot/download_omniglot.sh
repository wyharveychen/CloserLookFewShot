#!/usr/bin/env bash
wget https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/omniglot/splits/vinyals/train.txt 
wget https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/omniglot/splits/vinyals/val.txt
wget https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/omniglot/splits/vinyals/test.txt

DATADIR=./images
mkdir -p $DATADIR
wget -O images_background.zip https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip?raw=true
wget -O images_evaluation.zip https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip?raw=true
unzip images_background.zip -d $DATADIR
unzip images_evaluation.zip -d $DATADIR
mv $DATADIR/images_background/* $DATADIR/
mv $DATADIR/images_evaluation/* $DATADIR/
rmdir $DATADIR/images_background
rmdir $DATADIR/images_evaluation

python rot_omniglot.py
python write_omniglot_filelist.py
python write_cross_char_base_filelist.py
