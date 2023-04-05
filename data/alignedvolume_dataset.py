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


#TODO this dataset mode is WRONG. Figure out what to do.
class AlignedVolumeDataset(BaseDataset):
    """
    Loads image volume dataset. The dataset is consisted of multiple 3D image sub-volumes.
    This dataset loads one volume each from the source and target datasets.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=False):
        parser.add_argument('--data_ref', help = 'path to reference images')
        parser.add_argument('--validate', action= 'store_true', help = 'select whether you want to get validation loss while training')

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

        self.validate = opt.validate

        btoA = self.opt.direction == 'BtoA'

        self.isTrain = opt.isTrain

    def __getitem__(self, index):
        # apply image transformation

        transform_A = get_transform(self.opt)
        transform_B = get_transform(self.opt)

        A = transform_A(self.A_img_np)
        B = transform_B(self.B_img_np)


        if self.validate:
            transform_params_val = get_params(self.opt, self.A_img_shape)
            transform_val = get_transform(self.opt, params = transform_params_val)
            C = transform_val(self.A_img_np)
            D = transform_val(self.B_img_np)

            return {'src': A, 'src_paths': self.A_path, 'tgt': B, 'tgt_paths': self.B_path, 'src_val': C, 'tgt_val': D}

        else:
            return {'src': A, 'src_paths': self.A_path, 'tgt': B, 'tgt_paths': self.B_path}

    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        """

        # each epoch is 10 images.
        return int(10)