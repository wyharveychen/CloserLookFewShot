from __future__ import print_function

import os
import os.path
import numpy as np
import pickle



from PIL import Image

# Set the appropriate paths of the datasets here.
_CIFAR_FS_DATASET_DIR = 'Cifar100BySuperclass/'

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

def split_in_superclasses():
    d1 = load_data(os.path.join(
            _CIFAR_FS_DATASET_DIR,
            'cifar-100-python/test'))
    d2 = load_data(os.path.join(
        _CIFAR_FS_DATASET_DIR,
        'cifar-100-python/train'))
    meta = load_data(os.path.join(
        _CIFAR_FS_DATASET_DIR,
        'cifar-100-python/meta'))

    number_of_superclasses = 20
    fine_labels = np.concatenate((np.array(d1['fine_labels']), np.array(d2['fine_labels'])))
    data = np.concatenate((np.array(d1['data']), np.array(d2['data'])))
    coarse_labels = np.concatenate((np.array(d1['coarse_labels']), np.array(d2['coarse_labels'])))
    for i in range(number_of_superclasses):
        superclass_mask = coarse_labels == i
        y_of_superclass = fine_labels[superclass_mask]
        assert len(y_of_superclass) == 3000 # 5 * 600
        x_of_superclass = data[superclass_mask, :]
        superclass_name = meta["coarse_label_names"][i]
        with open(f"superclass_{superclass_name}.pickle", 'wb') as f:
            pickle.dump({
                "superclass": superclass_name,
                "labels": y_of_superclass,
                "data": x_of_superclass
            }, f)

