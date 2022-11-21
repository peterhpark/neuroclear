import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from skimage import io
import numpy as np
import re


def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


class DoubleCubeDataset(BaseDataset):
    """
    Loads image volume dataset. The dataset is consisted of multiple 3D image sub-volumes.
    This dataset loads one volume each from the source and target datasets.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=False):
        parser.add_argument('--dataref', help = 'path to reference images')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseDataset.__init__(self, opt)
        self.src_paths = make_dataset(opt.dataroot)
        self.tgt_paths = make_dataset(opt.dataref)
        self.src_paths.sort(key = numericalSort)
        self.tgt_paths.sort(key = numericalSort)

        self.src_size = len(self.src_paths)  # get the size of dataset A
        self.tgt_size = len(self.tgt_paths) # get the size of dataset B

        btoA = self.opt.direction == 'BtoA'
        self.transform_A = get_transform(self.opt)
        self.transform_B = get_transform(self.opt)

        self.isTrain = opt.isTrain

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, src_paths and tgt_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            src_paths (str)    -- image paths
            tgt_paths (str)    -- image paths
        """

        src_path = self.src_paths[index % self.src_size]  # make sure index is within then range
        tgt_path = self.tgt_paths[index % self.tgt_size]

        # Switch to importing a numpy file.
        src_img_np = io.imread(src_path)
        tgt_img_np = io.imread(tgt_path)

        # apply image transformation
        A = self.transform_A(src_img_np)
        B = self.transform_B(tgt_img_np)

        return {'src': A, 'src_paths': src_path, 'tgt': B, 'tgt_paths': tgt_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of the two datasets.
        """
        return np.max(self.src_size, self.tgt_size)
