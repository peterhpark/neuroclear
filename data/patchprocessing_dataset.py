from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from util.util import normalize
from skimage import io
from util import util
import numpy as np
import torchio as tio 

class PatchProcessingDataset(BaseDataset):
    """
    This dataset class uses torchio's grid sampler to process the image volume in any patch size (even asymmetric) in patches. 

    """

    @staticmethod
    def modify_commandline_options(parser, is_train=False):
        parser.add_argument('--patch_size', type=int, default=[120,120,120], help = 'set the size of the patch.')
        parser.add_argument('--overlap', type=int, default=0, help = 'set the size of overlapping region when dicing the dataset.')
        return parser
    
    def __init__(self, opt):
        """Initialize this dataset class.
        
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """ 

        BaseDataset.__init__(self, opt)
        self.A_path = make_dataset(opt.data_source, 1)[0]
        self.roi_size = opt.dice_size
        self.overlap = opt.overlap