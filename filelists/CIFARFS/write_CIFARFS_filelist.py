from os import listdir
from os.path import isdir, join
import os
from PIL import Image
import pickle
from pathlib import Path
from collections import defaultdict

base_path = os.getcwd()
file_list_path = join(base_path, "CifarFS")
data_path = join(base_path,'Cifar100BySuperclass/')
savedir = './'
dataset_list = ['base','val','novel']


def load_data(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data

# 1. Write pickled images to files (Structure: Superclass >> Class >> Images.jpg
fine_label_names = load_data(join(base_path, 'meta'))["fine_label_names"]
assert len(fine_label_names) == 100
all_pickled_files = [f for f in listdir(data_path) if ".pickle" in f]
assert len(all_pickled_files) == 20

class_name_to_label = {}
class_name_to_path = {}
for pickled_file in all_pickled_files:
    superclass_data = load_data(join(data_path, pickled_file))
    superclass_name = superclass_data["superclass"]
    labels = superclass_data["labels"]
    data = superclass_data["data"]
    superclass_path = join(file_list_path, superclass_name)
    Path(superclass_path).mkdir(parents=True, exist_ok=True)
    num_data_points_per_class = defaultdict(int)
    for label, data_point in zip(labels, data):
        num_data_points_per_class[label] += 1
        class_name = fine_label_names[label]
        class_path = join(superclass_path, class_name)
        class_name_to_label[class_name] = label
        class_name_to_path[class_name] = class_path
        Path(class_path).mkdir(parents=True, exist_ok=True)
        im = Image.fromarray(data_point.reshape(3, 32, 32).transpose(1, 2, 0))
        im.save(join(class_path, f"{class_name}_{num_data_points_per_class[label]}.jpg"))

with open(f"class_name_to_label.pickle", 'wb') as f:
    pickle.dump(class_name_to_label, f)

with open(f"class_name_to_path.pickle", 'wb') as f:
    pickle.dump(class_name_to_path, f)