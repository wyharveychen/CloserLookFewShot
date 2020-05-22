import pickle
import os
from os import listdir
from os.path import isfile, isdir, join
from collections import defaultdict
import json


base_path = os.getcwd()
cifar_path = join(base_path, "CloserLookFewShot/filelists/CIFARFS/")
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
    with open(join(cifar_path, f"{name}.json"), 'w') as f:
        json.dump(d, f)

def check_input(class_list, all_classes):
    assert all([cls in all_classes for cls in class_list]), f"Only classes for {all_classes} are allowed"


def generate_cross_domain_split(base_classes, val_classes, novel_classes):
    """
    Generate base, val and novel json-files on the fly.
    :param base_classes: the list of classes to be used for training
    :param val_classes: the list of classes to be used for validation
    :param novel_classes: the list of classes to be used for testing
    :return:
    """
    with open(join(cifar_path, "class_name_to_label.pickle"), 'rb') as f:
        class_name_to_label = pickle.load(f)
    with open(join(cifar_path, "class_name_to_path.pickle"), 'rb') as f:
        class_name_to_path = pickle.load(f)
    class_list = list(class_name_to_path.keys())
    check_input(base_classes, class_list)
    check_input(val_classes, class_list)
    check_input(novel_classes, class_list)
    write_json(base_classes, "base", class_name_to_path, class_name_to_label)
    write_json(val_classes, "val", class_name_to_path, class_name_to_label)
    write_json(novel_classes, "novel", class_name_to_path, class_name_to_label)