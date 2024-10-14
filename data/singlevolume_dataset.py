import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from skimage import io
import random
import re
from data.base_dataset import rotate_clean_3D_xy

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
        self.img_path = make_dataset(opt.dataroot, 1)[0] # loads only one image volume.
        self.img_vol = io.imread(self.img_path)
        self.aug_rotate_freq = opt.aug_rotate_freq

        btoA = self.opt.direction == 'BtoA'
        self.transform_A = get_transform(self.opt)

        self.img_vol_rotated = self.img_vol # initialize it as not rotated. 
        self.isTrain = opt.isTrain

    def __getitem__(self, index):

        # apply image transformation
        # iter_index = self.dummy_list[index]

        if index % self.aug_rotate_freq == 0: 
            angle = random.randint(0, 359)
            self.img_vol_rotated = rotate_clean_3D_xy(self.img_vol, angle) # 3D rotate at a random angle 
        
        A = self.transform_A(self.img_vol_rotated)
        return {'A': A, 'A_paths': self.img_path}
        
    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of the two datasets.
        """

        # each epoch is 10 images.
        return int(10)