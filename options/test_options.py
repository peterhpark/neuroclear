# TODO Sept 08 version
from .base_options import BaseOptions
import os

class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--dataroot_gt', help='path to images for comparison (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavior during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=3000, help='how many test images to run')
        parser.add_argument('--data_name', type=str, default=None, help='the name of the dataset that you run inference on.')
        parser.add_argument('--data_type', type=str, default='uint16', help='specify the data type for the output.')
        parser.add_argument('--histogram_match', action='store_true', default = False, help='Do histogram matching with an input sub-volume after inference.')
        parser.add_argument('--normalize_intensity', action='store_true', default = False, help='Perform intensity normalization after inference based on histogram.')
        parser.add_argument('--sat_level', type = float, nargs = '+', default = [0.25, 99.75], help='Set saturation levels for intensity normalization.')


        parser.add_argument('--background_threshold', type = float, nargs = '+', default = [None, None], help='Set the threshold for the background: e.g. background_value threshold: 2570 14000')
        parser.add_argument('--reference_slice_range', type = int, nargs = '+', default = [None, None], help='Set the slice range for calculating metrics.')

        # parser.add_argument('--adaptive_histogram', action='store_true', help='use adaptive histogram for visualization.')
        parser.add_argument('--save_slices', action='store_true', help='save sliced images (in 2D). ')
        parser.add_argument('--save_volume', action='store_true', help='save image volumes (in 3D). ')
        parser.add_argument('--save_projections', action='store_true', help='save MIP images (in 2D). ')

        parser.add_argument('--compare_with_gt', action='store_true', help='load the Ground-truth and compute metrics. ')
        parser.add_argument('--repetition', action='store_true', help='use redundancy in inference to reduce the output variations. ')
        parser.add_argument('--skip_real', action='store_true', help='Skip saving input image files. ')

        # parser.add_argument('--kernel_space', default = 41, type = int, help = 'overall domain space for convolution')
        # parser.add_argument('--kernel_size', default = 0, type = int, help ='size of the cropped kernel for convolution (for comparison with ground-truth kernel')


        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser