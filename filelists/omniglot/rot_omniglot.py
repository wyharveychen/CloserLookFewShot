import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random
from PIL import Image

cwd = os.getcwd() 
data_path = join(cwd,'images')
savedir = './'

#if not os.path.exists(savedir):
#    os.makedirs(savedir)

language_folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
language_folder_list.sort()

classfile_list_all = []

for language_folder in language_folder_list:
    language_folder_path = join(data_path, language_folder)
    character_folder_list = [cf for cf in listdir(language_folder_path) if isdir(join(language_folder_path, cf))]
    character_folder_list.sort()
    for character_folder in character_folder_list:
        character_folder_path = join(language_folder_path, character_folder)        
        image_list =  [ img for img in listdir(character_folder_path) if (isfile(join(character_folder_path,img)) and img[0] != '.')]
        for deg in [0,90,180,270]:
            rot_str = "rot%03d"%deg
            rot_character_path = join(character_folder_path, rot_str)
            print(rot_character_path)
            if not os.path.exists(rot_character_path):
                os.makedirs(rot_character_path)
            for img in image_list:                
                rot_img = Image.open(join(character_folder_path,img)).rotate(deg)
                rot_img.save(join(character_folder_path,rot_str,img))

