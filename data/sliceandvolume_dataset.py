from data.base_dataset import BaseDataset, get_transform, get_params
from options.train_options import TrainOptions
from data.image_folder import make_dataset
from skimage import io
import re
import util.util as util
import numpy as np

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


class SliceAndVolumeDataset(BaseDataset):
    """
    Loads image volume dataset. The dataset is consisted of one 3D image volume and multiple 2D images
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=False):
        parser.add_argument('--data_ref', help = 'path to reference images')
        parser.add_argument('--data_gt', type=str, default=None, help='specify the path to the groundtruth')
        parser.add_argument('--targetsizing', type=int, default=1, help='specify how big to make (xTimes) the crop size in the target domain for 2D discriminator')

        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseDataset.__init__(self, opt)
        self.A_path = make_dataset(opt.dataroot, 1)[0]  # loads only one 3D image.
        self.A_img_np = io.imread(self.A_path)
        self.A_img_shape = self.A_img_np.shape

        self.B_paths = make_dataset(opt.data_ref, 100)  # loads multiple 2D images
        self.B_size = len(self.B_paths)

        self.validate = False
        if opt.data_gt is not None:
            self.validate = True
            self.C_path = make_dataset(opt.data_gt, 1)[0] # loads only one 3D image.
            self.C_img_np = io.imread(self.C_path)

        btoA = self.opt.direction == 'BtoA'

        self.isTrain = opt.isTrain

    def __getitem__(self, index):

        B_path = self.B_paths[index % self.B_size]  # make sure index is within then range

        # Switch to importing a numpy file.
        B_img_np = io.imread(B_path)

        # apply image transformation
        transform_A = get_transform(self.opt)
        transform_B = get_transform(self.opt, is_2D=True) # randomize separately

        A = transform_A(self.A_img_np)
        B = transform_B(B_img_np)

        # A_np = util.tensor2im(A, imtype=np.uint8).squeeze()[:,20]
        # B_np = util.tensor2im(B, imtype=np.uint8).squeeze()

        if self.validate:
            C = transform_A(self.C_img_np)
            return {'src': A, 'src_paths': self.A_path, 'tgt': B, 'tgt_paths': B_path, 'gt': C, 'gt_paths': self.C_path}

        else:
            return {'src': A, 'src_paths': self.A_path, 'tgt': B, 'tgt_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        """

        # each epoch is 100 images.
        return int(100)