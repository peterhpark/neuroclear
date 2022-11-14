import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from skimage import io
import re


def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


class DoubleCubeDataset(BaseDataset):
    """
    Loads image volume dataset. The dataset is consisted of multiple 3D image sub-volumes.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseDataset.__init__(self, opt)
        self.A_paths = make_dataset(opt.dataroot)
        self.B_paths = make_dataset(opt.)
        self.A_paths.sort(key = numericalSort)

        self.A_size = len(self.A_paths)  # get the size of dataset A

        btoA = self.opt.direction == 'BtoA'
        self.transform_A = get_transform(self.opt)

        self.isTrain = opt.isTrain

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """

        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range

        # Switch to importing a numpy file.
        A_img_np = io.imread(A_path)

        # apply image transformation
        A = self.transform_A(A_img_np)

        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of the two datasets.
        """
        return self.A_size
