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

    @staticmethod
    def modify_commandline_options(parser, isTrain=True):

        parser.add_argument('--get_img_params', action='store_true', default=False,
                            help='if true, calculate the mean and std. from the loaded image volume.')

        opt, _ = parser.parse_known_args()
        if opt.get_img_params:
            file_path = make_dataset(opt.dataroot, 1)[0]  # loads only one image volume.
            img_vol = io.imread(file_path)
            img_params = (np.mean(img_vol), np.std(img_vol))
            img_params = img_params[0].astype(float), img_params[1].astype(float)
            if img_vol.dtype == np.uint8:
                img_params = img_params[0] / (2 ** 8 * 1.0 - 1), img_params[1] / (2 ** 8 * 1.0 - 1)
            elif img_vol.dtype == np.uint16:
                img_params = img_params[0] / (2 ** 16 * 1.0 - 1), img_params[1] / (2 ** 16 * 1.0 - 1)
            parser.set_defaults(img_params=img_params)

            print ("-----------------NOTE------------------")
            print ("Image parameters are calculated in dataset module.")
            print ("-> image mean: %f, image std: %f" %(img_params[0], img_params[1]))
            print ("---------------------------------------")

        parser.add_argument('--epoch_length', type=int, default=2000, help = 'Set how many iterations per epoch.')
        return parser


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
        # self.dummy_list = [x for x in range(opt.epoch_length)] # create a dummy list for iteration
        # self.iteration_size = len(self.dummy_list)
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