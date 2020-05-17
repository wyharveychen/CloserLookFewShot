import random
import numpy as np
import pickle
import os
from os import listdir
from os.path import isfile, isdir, join
from collections import defaultdict
import json


base_path = os.getcwd()

def write_json(class_list, name, class_name_to_path, class_name_to_label):
    d = defaultdict(list)
    d["label_names"] = list(class_name_to_path.keys())
    for base_class in class_list:
        path = class_name_to_path[base_class]
        label = class_name_to_label[base_class]
        for file in listdir(path):
            if ".jpg" in file:
                d["image_labels"].append(int(label))
                d["image_names"].append(join(path, file))
    with open(f"{name}.json", 'w') as f:
        json.dump(d, f)


def generate_cross_domain_split(base_classes, val_classes, novel_classes):
    """
    Generate base, val and novel json-files on the fly.
    :param base_classes: the list of classes to be used for training
    :param val_classes: the list of classes to be used for validation
    :param novel_classes: the list of classes to be used for testing
    :return:
    """
    with open("class_name_to_label.pickle", 'rb') as f:
        class_name_to_label = pickle.load(f)
    with open("class_name_to_path.pickle", 'rb') as f:
        class_name_to_path = pickle.load(f)
    write_json(base_classes, "base", class_name_to_path, class_name_to_label)
    write_json(val_classes, "val", class_name_to_path, class_name_to_label)
    write_json(novel_classes, "novel", class_name_to_path, class_name_to_label)