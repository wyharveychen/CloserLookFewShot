import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random
from PIL import Image
import PIL.ImageOps

cwd = os.getcwd() 
data_path = join(cwd,'emnist')
inv_data_path = join(cwd,'inv_emnist')
savedir = './'

#if not os.path.exists(savedir):
#    os.makedirs(savedir)
if not os.path.exists(inv_data_path):
    os.makedirs(inv_data_path)

character_folder_list = [str(i) for i in range(62)] #lazy_hack 

classfile_list_all = []

for character_folder in character_folder_list:
    character_folder_path = join(data_path, character_folder)        
    inv_character_folder_path = join(inv_data_path, character_folder)        
    image_list =  [ img for img in listdir(character_folder_path) if (isfile(join(character_folder_path,img)) and img[0] != '.')]
    if not os.path.exists(inv_character_folder_path):
        os.makedirs(inv_character_folder_path)
    for img in image_list:                
        inverted_img =PIL.ImageOps.invert(Image.open(join(character_folder_path,img)))
        inverted_img.save(join(inv_character_folder_path ,img))

