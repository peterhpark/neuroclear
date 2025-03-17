from data.base_dataset import BaseDataset, get_transform, get_params
from options.train_options import TrainOptions
from data.image_folder import make_dataset
from skimage import io
import re
import util.util as util
from data.base_dataset import rotate_clean_3D_xy
from data.base_dataset import rotate_clean_2D
import random 
import sys

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


class SliceAndVolumeDataset(BaseDataset):
    """
    Loads image volume dataset. The dataset is consisted of one 3D image volume (source) and multiple 2D images (target)
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=False):
        parser.add_argument('--data_target', help = 'path to reference images')
        parser.add_argument('--data_gt', type=str, default=None, help='specify the path to the groundtruth')
        parser.add_argument('--crop_size_2D', type=int, default=256, help='crop size for 2D images')
        parser.add_argument('--crop_size_3D', type=int, default=64, help='crop size for 3D images')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        #FIXME for domain A, we only load one volume always. 
        BaseDataset.__init__(self, opt)
        self.A_paths = make_dataset(opt.data_source)  
        self.aug_rotate_freq = opt.aug_rotate_freq
        self.A_size = len(self.A_paths)

        if self.A_size <=10: # if the image files are less than 10 files we load all the images in memory.
            self.A_imgs = [io.imread(A_path) for A_path in self.A_paths]
            self.A_loaded_in_memory = True
            total_size_A = sys.getsizeof(self.A_imgs ) + sum(sys.getsizeof(item) for item in self.A_imgs)
            print(f"Total memory size loaded for source images : {total_size_A} bytes")
        else:
            self.A_loaded_in_memory = False

            
        self.B_paths = make_dataset(opt.data_target)  # loads multiple 2D images
        self.B_size = len(self.B_paths)
        
        if self.B_size <=10: # if the image files are less than 10 files we load all the images in memory.
            self.B_imgs = [io.imread(B_path) for B_path in self.B_paths]
            self.B_loaded_in_memory = True
            total_size_B = sys.getsizeof(self.B_imgs) + sum(sys.getsizeof(item) for item in self.B_imgs)
            print(f"Total memory size loaded for target images : {total_size_B} bytes")

            total_size_B = sum(sys.getsizeof(item) for item in self.B_imgs)
            print(f"Total memory size loaded for target images : {total_size_B} bytes")
        else:
            self.B_loaded_in_memory = False

        # Check if the first image in B_paths has 3 dimensions
        self.first_B_image_vol = io.imread(self.B_paths[0])
        self.B_is_3D = self.first_B_image_vol.ndim == 3

        self.validate = False
        self.crop_size_2D = opt.crop_size_2D
        self.crop_size_3D = opt.crop_size_3D

        self.transform_A = get_transform(self.opt, crop_size=self.crop_size_3D, is_2D=False)
        self.transform_B = get_transform(self.opt, crop_size=self.crop_size_2D, is_2D=True) # apply a different crop size for 2D images.


        # Load the ground truth data if it is provided.
        if hasattr(opt, 'data_gt'):
            self.validate = True
            self.C_path = make_dataset(opt.data_gt, 1)[0] # loads only one 3D image.
            self.C_img_np = io.imread(self.C_path)

        btoA = self.opt.direction == 'BtoA'

        self.isTrain = opt.isTrain

    def __getitem__(self, index):
        '''
        Samples for a source image volume and a target image slice. 

        We do 2D/3D rotation for data augmentation here. 
        '''
        A_path = self.A_paths[index % self.A_size]

        if self.A_loaded_in_memory == False:
            A_img_vol = io.imread(A_path)
        else:
            A_img_vol = self.A_imgs[index % self.A_size]

        angle_A = self.sample_angle()
        A_img_vol = rotate_clean_3D_xy(A_img_vol, angle_A) # 3D rotate at a random angle. 

        B_path = self.B_paths[index % self.B_size]  # make sure index is within the range
        if self.B_loaded_in_memory == False:
            B_img = io.imread(B_path)
        else:
            B_img = self.B_imgs[index % self.B_size]

        # If the image is 3D, we need to select a slice.
        if self.B_is_3D:
            slice_index = random.randint(0, B_img.shape[0] - 1)
            B_img_slice = B_img[slice_index, :, :]
            
        # If the image is 2D, we can use the image as is.
        else:
            B_img_slice = B_img

        angle_B = self.sample_angle()
        B_img = rotate_clean_2D(B_img_slice, angle_B)

        # apply image transformation
        A = self.transform_A(A_img_vol)
        slice_A = A[:,:,10,:,:]
        B = self.transform_B(B_img_slice)

        if self.validate:
            C = self.transform_A(self.C_img_np)
            return {'src': A, 'src_paths': A_path, 'tgt': B, 'tgt_paths': B_path, 'gt': C, 'gt_paths': self.C_path}

        else:
            return {'src': A, 'src_paths': A_path, 'tgt': B, 'tgt_paths': B_path}

    @staticmethod
    def sample_angle():
        # Rotate the 3D source image volume every aug_rotate_freq iterations.
        if random.random() < 0.8:  # sample more at this range of angles because it crops out less.
            base_angle = random.choice([0, 90, 180, 270])
            angle = base_angle + random.uniform(-15, 15)
        else:
            angle = random.randint(0, 359)
        return angle
    
    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        """

        # each epoch is 100 images.
        return int(5000)