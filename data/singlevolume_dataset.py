#TODO SEP 08 version
import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from skimage import io
from data.image_folder import merge_datasets
import random
import numpy as np
import re


def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts



class SingleVolumeDataset(BaseDataset):
    """
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseDataset.__init__(self, opt)
        self.A_path = make_dataset(opt.dataroot, 1)[0] # loads only one image volume.
        self.A_img_np = io.imread(self.A_path)

        self.img_params = opt.img_params
        btoA = self.opt.direction == 'BtoA'
        self.transform_A = get_transform(self.opt)
        self.isTrain = opt.isTrain


    def __getitem__(self, index):

        # apply image transformation
        # iter_index = self.dummy_list[index]
        A = self.transform_A(self.A_img_np)
        return {'A': A, 'A_paths': self.A_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of the two datasets.
        """

        # each epoch is 10 images.
        return int(10)