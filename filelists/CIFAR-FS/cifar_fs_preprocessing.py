# Dataloader of Gidaris & Komodakis, CVPR 2018
# Adapted from:
# https://github.com/gidariss/FewShotWithoutForgetting/blob/master/dataloader.py
from __future__ import print_function

import os
import os.path
import numpy as np
import random
import pickle

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchnet as tnt


from PIL import Image

# Set the appropriate paths of the datasets here.
_CIFAR_FS_DATASET_DIR = 'Cifar100BySuperclass/'


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


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


class CIFAR_FS(data.Dataset):
    def __init__(self, phase='train', do_not_use_random_transf=False):

        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.name = 'CIFAR_FS_' + phase

        print('Loading CIFAR-FS dataset - phase {0}'.format(phase))
        file_train_categories_train_phase = os.path.join(
            _CIFAR_FS_DATASET_DIR,
            'CIFAR_FS_train.pickle')
        file_train_categories_val_phase = os.path.join(
            _CIFAR_FS_DATASET_DIR,
            'CIFAR_FS_train.pickle')
        file_train_categories_test_phase = os.path.join(
            _CIFAR_FS_DATASET_DIR,
            'CIFAR_FS_train.pickle')
        file_val_categories_val_phase = os.path.join(
            _CIFAR_FS_DATASET_DIR,
            'CIFAR_FS_val.pickle')
        file_test_categories_test_phase = os.path.join(
            _CIFAR_FS_DATASET_DIR,
            'CIFAR_FS_test.pickle')

        if self.phase == 'train':
            # During training phase we only load the training phase images
            # of the training categories (aka base categories).
            data_train = load_data(file_train_categories_train_phase)
            self.data = data_train['data']
            self.labels = data_train['labels']

            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = self.labelIds
            self.num_cats_base = len(self.labelIds_base)

        elif self.phase == 'val' or self.phase == 'test':
            if self.phase == 'test':
                # load data that will be used for evaluating the recognition
                # accuracy of the base categories.
                data_base = load_data(file_train_categories_test_phase)
                # load data that will be use for evaluating the few-shot recogniton
                # accuracy on the novel categories.
                data_novel = load_data(file_test_categories_test_phase)
            else:  # phase=='val'
                # load data that will be used for evaluating the recognition
                # accuracy of the base categories.
                data_base = load_data(file_train_categories_val_phase)
                # load data that will be use for evaluating the few-shot recogniton
                # accuracy on the novel categories.
                data_novel = load_data(file_val_categories_val_phase)

            self.data = np.concatenate(
                [data_base['data'], data_novel['data']], axis=0)
            self.labels = data_base['labels'] + data_novel['labels']

            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)

            self.labelIds_base = buildLabelIndex(data_base['labels']).keys()
            self.labelIds_novel = buildLabelIndex(data_novel['labels']).keys()
            self.num_cats_base = len(self.labelIds_base)
            self.num_cats_novel = len(self.labelIds_novel)
            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert (len(intersection) == 0)
        else:
            raise ValueError('Not valid phase {0}'.format(self.phase))

        mean_pix = [x / 255.0 for x in [129.37731888, 124.10583864, 112.47758569]]

        std_pix = [x / 255.0 for x in [68.20947949, 65.43124043, 70.45866994]]

        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        if (self.phase == 'test' or self.phase == 'val') or (do_not_use_random_transf == True):

            self.transform = transforms.Compose([
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])
        else:

            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

class CIFARFSDomainShift(data.Dataset):

    def __init__(self, superclass_train, super_class_test, phase="train"):
        self.phase = phase
        train = load_data(os.path.join(
            _CIFAR_FS_DATASET_DIR,
            f'superclass_{superclass_train}.pickle'))
        if superclass_train != super_class_test:
            test = load_data(os.path.join(
                _CIFAR_FS_DATASET_DIR,
                f'superclass_{superclass_train}.pickle'))
            self.x_test = test["data"]
            self.y_test = test["labels"]
            self.x_train = train["data"]
            self.y_train = train["labels"]
        else:
            labels = train["labels"]
            data = train["data"]
            random_class = labels[np.random.choice(labels.shape[0], 1, replace=False)]
            self.x_test = labels[labels==random_class]
            self.y_test = data[labels==random_class, :]
            self.x_train = labels[labels!=random_class]
            self.y_train = data[labels!=random_class, :]


    def set_phase(self, phase):
        assert(phase == 'train' or phase == 'test')
        self.phase = phase

    def __getitem__(self, index):
        if self.phase == "train":
            img, label = self.x_train[index], self.y_train[index]
        elif self.phase == "test":
            img, label = self.x_test[index], self.y_test[index]
        else:
            raise ValueError(f"phase can be either train or test but is {self.phase}")
        img = Image.fromarray(img)
        return img, label

    def __len__(self):
        if self.phase == "train":
            return len(self.y_train)
        elif self.phase == "test":
            return len(self.y_test)
        else:
            raise ValueError(f"phase can be either train or test but is {self.phase}")

class FewShotDataloader():
    def __init__(self,
                 dataset,
                 nKnovel=5,  # number of novel categories.
                 nKbase=-1,  # number of base categories.
                 nExemplars=1,  # number of training examples per novel category.
                 nTestNovel=15 * 5,  # number of test examples for all the novel categories.
                 nTestBase=15 * 5,  # number of test examples for all the base categories.
                 batch_size=1,  # number of training episodes per batch.
                 num_workers=4,
                 epoch_size=2000,  # number of batches per epoch.
                 ):

        self.dataset = dataset
        self.phase = self.dataset.phase
        max_possible_nKnovel = (self.dataset.num_cats_base if self.phase == 'train'
                                else self.dataset.num_cats_novel)
        assert (nKnovel >= 0 and nKnovel < max_possible_nKnovel)
        self.nKnovel = nKnovel

        max_possible_nKbase = self.dataset.num_cats_base
        nKbase = nKbase if nKbase >= 0 else max_possible_nKbase
        if self.phase == 'train' and nKbase > 0:
            nKbase -= self.nKnovel
            max_possible_nKbase -= self.nKnovel

        assert (nKbase >= 0 and nKbase <= max_possible_nKbase)
        self.nKbase = nKbase

        self.nExemplars = nExemplars
        self.nTestNovel = nTestNovel
        self.nTestBase = nTestBase
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_workers = num_workers
        self.is_eval_mode = (self.phase == 'test') or (self.phase == 'val')

    def sampleImageIdsFrom(self, cat_id, sample_size=1):
        """
        Samples `sample_size` number of unique image ids picked from the
        category `cat_id` (i.e., self.dataset.label2ind[cat_id]).
        Args:
            cat_id: a scalar with the id of the category from which images will
                be sampled.
            sample_size: number of images that will be sampled.
        Returns:
            image_ids: a list of length `sample_size` with unique image ids.
        """
        assert (cat_id in self.dataset.label2ind)
        assert (len(self.dataset.label2ind[cat_id]) >= sample_size)
        # Note: random.sample samples elements without replacement.
        return random.sample(self.dataset.label2ind[cat_id], sample_size)

    def sampleCategories(self, cat_set, sample_size=1):
        """
        Samples `sample_size` number of unique categories picked from the
        `cat_set` set of categories. `cat_set` can be either 'base' or 'novel'.
        Args:
            cat_set: string that specifies the set of categories from which
                categories will be sampled.
            sample_size: number of categories that will be sampled.
        Returns:
            cat_ids: a list of length `sample_size` with unique category ids.
        """
        if cat_set == 'base':
            labelIds = self.dataset.labelIds_base
        elif cat_set == 'novel':
            labelIds = self.dataset.labelIds_novel
        else:
            raise ValueError('Not recognized category set {}'.format(cat_set))

        assert (len(labelIds) >= sample_size)
        # return sample_size unique categories chosen from labelIds set of
        # categories (that can be either self.labelIds_base or self.labelIds_novel)
        # Note: random.sample samples elements without replacement.
        return random.sample(labelIds, sample_size)

    def sample_base_and_novel_categories(self, nKbase, nKnovel):
        """
        Samples `nKbase` number of base categories and `nKnovel` number of novel
        categories.
        Args:
            nKbase: number of base categories
            nKnovel: number of novel categories
        Returns:
            Kbase: a list of length 'nKbase' with the ids of the sampled base
                categories.
            Knovel: a list of lenght 'nKnovel' with the ids of the sampled novel
                categories.
        """
        if self.is_eval_mode:
            assert (nKnovel <= self.dataset.num_cats_novel)
            # sample from the set of base categories 'nKbase' number of base
            # categories.
            Kbase = sorted(self.sampleCategories('base', nKbase))
            # sample from the set of novel categories 'nKnovel' number of novel
            # categories.
            Knovel = sorted(self.sampleCategories('novel', nKnovel))
        else:
            # sample from the set of base categories 'nKnovel' + 'nKbase' number
            # of categories.
            cats_ids = self.sampleCategories('base', nKnovel + nKbase)
            assert (len(cats_ids) == (nKnovel + nKbase))
            # Randomly pick 'nKnovel' number of fake novel categories and keep
            # the rest as base categories.
            random.shuffle(cats_ids)
            Knovel = sorted(cats_ids[:nKnovel])
            Kbase = sorted(cats_ids[nKnovel:])

        return Kbase, Knovel

    def sample_test_examples_for_base_categories(self, Kbase, nTestBase):
        """
        Sample `nTestBase` number of images from the `Kbase` categories.
        Args:
            Kbase: a list of length `nKbase` with the ids of the categories from
                where the images will be sampled.
            nTestBase: the total number of images that will be sampled.
        Returns:
            Tbase: a list of length `nTestBase` with 2-element tuples. The 1st
                element of each tuple is the image id that was sampled and the
                2nd elemend is its category label (which is in the range
                [0, len(Kbase)-1]).
        """
        Tbase = []
        if len(Kbase) > 0:
            # Sample for each base category a number images such that the total
            # number sampled images of all categories to be equal to `nTestBase`.
            KbaseIndices = np.random.choice(
                np.arange(len(Kbase)), size=nTestBase, replace=True)
            KbaseIndices, NumImagesPerCategory = np.unique(
                KbaseIndices, return_counts=True)

            for Kbase_idx, NumImages in zip(KbaseIndices, NumImagesPerCategory):
                imd_ids = self.sampleImageIdsFrom(
                    Kbase[Kbase_idx], sample_size=NumImages)
                Tbase += [(img_id, Kbase_idx) for img_id in imd_ids]

        assert (len(Tbase) == nTestBase)

        return Tbase

    def sample_train_and_test_examples_for_novel_categories(
            self, Knovel, nTestNovel, nExemplars, nKbase):
        """Samples train and test examples of the novel categories.
        Args:
    	    Knovel: a list with the ids of the novel categories.
            nTestNovel: the total number of test images that will be sampled
                from all the novel categories.
            nExemplars: the number of training examples per novel category that
                will be sampled.
            nKbase: the number of base categories. It is used as offset of the
                category index of each sampled image.
        Returns:
            Tnovel: a list of length `nTestNovel` with 2-element tuples. The
                1st element of each tuple is the image id that was sampled and
                the 2nd element is its category label (which is in the range
                [nKbase, nKbase + len(Knovel) - 1]).
            Exemplars: a list of length len(Knovel) * nExemplars of 2-element
                tuples. The 1st element of each tuple is the image id that was
                sampled and the 2nd element is its category label (which is in
                the ragne [nKbase, nKbase + len(Knovel) - 1]).
        """

        if len(Knovel) == 0:
            return [], []

        nKnovel = len(Knovel)
        Tnovel = []
        Exemplars = []
        assert ((nTestNovel % nKnovel) == 0)
        nEvalExamplesPerClass = int(nTestNovel / nKnovel)

        for Knovel_idx in range(len(Knovel)):
            imd_ids = self.sampleImageIdsFrom(
                Knovel[Knovel_idx],
                sample_size=(nEvalExamplesPerClass + nExemplars))

            imds_tnovel = imd_ids[:nEvalExamplesPerClass]
            imds_ememplars = imd_ids[nEvalExamplesPerClass:]

            Tnovel += [(img_id, nKbase + Knovel_idx) for img_id in imds_tnovel]
            Exemplars += [(img_id, nKbase + Knovel_idx) for img_id in imds_ememplars]
        assert (len(Tnovel) == nTestNovel)
        assert (len(Exemplars) == len(Knovel) * nExemplars)
        random.shuffle(Exemplars)

        return Tnovel, Exemplars

    def sample_episode(self):
        """Samples a training episode."""
        nKnovel = self.nKnovel
        nKbase = self.nKbase
        nTestNovel = self.nTestNovel
        nTestBase = self.nTestBase
        nExemplars = self.nExemplars

        Kbase, Knovel = self.sample_base_and_novel_categories(nKbase, nKnovel)
        Tbase = self.sample_test_examples_for_base_categories(Kbase, nTestBase)
        Tnovel, Exemplars = self.sample_train_and_test_examples_for_novel_categories(
            Knovel, nTestNovel, nExemplars, nKbase)

        # concatenate the base and novel category examples.
        Test = Tbase + Tnovel
        random.shuffle(Test)
        Kall = Kbase + Knovel

        return Exemplars, Test, Kall, nKbase

    def createExamplesTensorData(self, examples):
        """
        Creates the examples image and label tensor data.
        Args:
            examples: a list of 2-element tuples, each representing a
                train or test example. The 1st element of each tuple
                is the image id of the example and 2nd element is the
                category label of the example, which is in the range
                [0, nK - 1], where nK is the total number of categories
                (both novel and base).
        Returns:
            images: a tensor of shape [nExamples, Height, Width, 3] with the
                example images, where nExamples is the number of examples
                (i.e., nExamples = len(examples)).
            labels: a tensor of shape [nExamples] with the category label
                of each example.
        """
        images = torch.stack(
            [self.dataset[img_idx][0] for img_idx, _ in examples], dim=0)
        labels = torch.LongTensor([label for _, label in examples])
        return images, labels

    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        def load_function(iter_idx):
            Exemplars, Test, Kall, nKbase = self.sample_episode()
            Xt, Yt = self.createExamplesTensorData(Test)
            Kall = torch.LongTensor(Kall)
            if len(Exemplars) > 0:
                Xe, Ye = self.createExamplesTensorData(Exemplars)
                return Xe, Ye, Xt, Yt, Kall, nKbase
            else:
                return Xt, Yt, Kall, nKbase

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=load_function)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=(0 if self.is_eval_mode else self.num_workers),
            shuffle=(False if self.is_eval_mode else True))

        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return int(self.epoch_size / self.batch_size)

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

if __name__=="__main__":
    dataset = CIFARFSDomainShift("aquatic_mammals", "aquatic_mammals")
