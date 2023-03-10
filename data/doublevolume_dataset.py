from data.base_dataset import BaseDataset, get_transform, get_params
from options.train_options import TrainOptions
from data.image_folder import make_dataset
from skimage import io
import re


def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


class DoubleVolumeDataset(BaseDataset):
    """
    Loads image volume dataset. The dataset is consisted of multiple 3D image sub-volumes.
    This dataset loads one volume each from the source and target datasets.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=False):
        parser.add_argument('--data_ref', help = 'path to reference images')
        parser.add_argument('--data_gt', type=str, default=None, help='specify the path to the groundtruth')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseDataset.__init__(self, opt)
        self.A_path = make_dataset(opt.dataroot, 1)[0]  # loads only one image volume.
        self.A_img_np = io.imread(self.A_path)
        self.A_img_shape = self.A_img_np.shape

        self.B_path = make_dataset(opt.data_ref, 1)[0]  # loads only one image volume.
        self.B_img_np = io.imread(self.B_path)

        self.validate = False
        if opt.data_gt is not None:
            self.validate = True
            self.C_path = make_dataset(opt.data_gt, 1)[0] # loads only one image volume.
            self.C_img_np = io.imread(self.C_path)

        btoA = self.opt.direction == 'BtoA'

        self.isTrain = opt.isTrain

    def __getitem__(self, index):
        # apply image transformation


        transform_params = get_params(self.opt, self.A_img_shape)
        transform_A = get_transform(self.opt, params = transform_params)
        transform_B = get_transform(self.opt) # still randomize

        A = transform_A(self.A_img_np)
        B = transform_B(self.B_img_np)


        if self.validate:
            C = transform_A(self.C_img_np)
            return {'src': A, 'src_paths': self.A_path, 'tgt': B, 'tgt_paths': self.B_path, 'gt': C, 'gt_paths': self.C_path}

        else:
            return {'src': A, 'src_paths': self.A_path, 'tgt': B, 'tgt_paths': self.B_path}

    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        """

        # each epoch is 10 images.
        return int(10)