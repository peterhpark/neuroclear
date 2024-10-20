from data.base_dataset import BaseDataset, get_transform, get_params
from options.train_options import TrainOptions
from data.image_folder import make_dataset
from skimage import io
import re
from data.base_dataset import rotate_clean_3D_xy
import os 
import random 

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

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseDataset.__init__(self, opt)
        self.A_path = make_dataset(opt.data_source)[0]
        self.A_img_vol = io.imread(self.A_path)
        self.A_img_shape = self.A_img_vol.shape

        self.B_path = make_dataset(opt.data_target)[0]
        self.B_img_vol = io.imread(self.B_path)
        self.aug_rotate_freq = opt.aug_rotate_freq
        self.epoch_length = int(self.aug_rotate_freq * 2327 * 0.25) # how many iterations per epoch? 2327 is minimum iterations to cover all angles

        self.rotate3D = 'random3Drotate' in opt.preprocess
        if self.rotate3D:
            print ("The dataloader will apply 3D rotation as part of data augmentation. This will slow down the data loading.")
    
        self.validate = False
        # if opt.data_gt is not None:
        #     self.validate = True
        #     self.C_path = make_dataset(opt.data_gt, 1)[0] # loads only one image volume.
        #     self.C_img_np = io.imread(self.C_path)

        btoA = self.opt.direction == 'BtoA'

        self.A_img_vol_rotated = self.A_img_vol # initialize it as not rotated. 
        self.isTrain = opt.isTrain

    def __getitem__(self, index):
        # apply image transformation
        if self.rotate3D: 
            if index % self.aug_rotate_freq == 0: 
                angle = random.randint(0, 359) # to cover all angles we need to sample 2327 times (coupon collector's problem)
                self.A_img_vol_rotated = rotate_clean_3D_xy(self.A_img_vol, angle) # 3D rotate at a random angle 

        transform_A = get_transform(self.opt)
        transform_B = get_transform(self.opt) # still randomize

        A = transform_A(self.A_img_vol_rotated)
        B = transform_B(self.B_img_vol)

        if self.validate:
            C = transform_A(self.C_img_np)
            return {'src': A, 'src_paths': self.A_path, 'tgt': B, 'tgt_paths': self.B_path, 'gt': C, 'gt_paths': self.C_path}

        else:
            # from PIL import Image
            # test_img_A = A[0,0,50,:,:].cpu().float().numpy()*255.0
            # im = Image.fromarray(test_img_A).convert('RGB')
            # im.save("test_img_A.png")
            # test_img_B = B[0,0,50,:,:].cpu().float().numpy()*255.0
            # im2 = Image.fromarray(test_img_B).convert('RGB')
            # im2.save("test_img_B.png")
            return {'src': A, 'src_paths': self.A_path, 'tgt': B, 'tgt_paths': self.B_path}

    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        """
        return self.epoch_length