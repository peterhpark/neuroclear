from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from util.util import normalize
from skimage import io
from util import util
import numpy as np


class DiceImageDataSet(BaseDataset):
    """This dataset class loads a single 3D image volume and dice it into cubes. --dataroot /path/to/data.
    Each cube refers to one index of data in this dataset.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=False):
        parser.add_argument('--overlap', type=int, default=0, help = 'set the size of overlapping region when dicing the dataset.')
        parser.add_argument('--border_cut', default =0, type = int, help = 'specify how much border you want to remove in a cube-by-cube inference.')

        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_path = make_dataset(opt.dataroot, 1)[0] # loads only one image volume.
        self.roi_size = opt.dice_size[0]
        self.overlap = opt.overlap
        self.border_cut = opt.border_cut

        A_img_np = io.imread(self.A_path)
        # A_img_np = np.load(self.A_path)

        # norm_parms = {'min_max':(np.min(A_img_np), np.max(A_img_np))}
        # self.transform = get_transform(opt, norm_parms)

        self.transform = get_transform(opt)

        self.image_size_original = A_img_np.shape
        # A_img_np = self.pad(A_img_np, self.roi_size, overlap=self.overlap) # pad the input image volume so that it gets diced cleanly (with no remainders).
        # A_img_np = util.crop_for_dicing(A_img_np, self.roi_size, overlap=self.overlap)

        A_img_np = util.pad_for_dicing(A_img_np, self.roi_size, overlap=self.overlap)
        self.image_size = A_img_np.shape
        self.cube = DiceCube(A_img_np, self.roi_size, overlap=self.overlap, border_cut = self.border_cut)


    def __getitem__(self, index): # index is for each image file.
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        cube = self.cube[index]
        A = self.transform(cube)

        return {'A': A, 'A_paths': str(index)}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.cube)

    def shape(self):
        return (self.cube.z_steps, self.cube.y_steps, self.cube.x_steps)

    def size(self):
        return  self.image_size

    def size_original(self):
        return self.image_size_original

class DiceCube():
    def __init__(self, image, roi_size, overlap = 0, border_cut = 0):
        self.image = image
        self.roi_size = roi_size
        self.overlap = overlap
        self.size = image.size
        self.step = self.roi_size - self.overlap
        self.border_cut = border_cut

        self.z_steps  = (self.image.shape[0]-self.overlap)//self.step
        self.y_steps = (self.image.shape[1]-self.overlap)//self.step
        self.x_steps  = (self.image.shape[2]-self.overlap)//self.step

        # Pad to deal with the indexing when having a border cut.
        npad = ((border_cut, border_cut), (border_cut, border_cut), (border_cut, border_cut))
        self.image = np.pad(self.image, pad_width=npad, mode='reflect')


    def indexToCoordinates(self, index): # converts 1D dicing order to 3D stacking order

        # Dicing order: x-> y-> z
        x_index = index % self.x_steps
        y_index = (index % (self.x_steps*self.y_steps))//self.x_steps
        z_index = (index) // (self.x_steps*self.y_steps)

        return z_index, y_index, x_index

    def __getitem__(self, index):
        self.index = index
        z_index, y_index, x_index = self.indexToCoordinates(index)
        current_y = y_index*(self.step) + self.border_cut
        current_x = x_index*(self.step) + self.border_cut
        current_z = z_index*(self.step) + self.border_cut

        # new_cube = self.image[current_z:current_z+self.roi_size, current_y:current_y+self.roi_size, current_x:current_x+self.roi_size]
        # overestimate the cube region with added borders, which will be later removed in Assemble_Dice.
        new_cube = self.image[current_z-self.border_cut:current_z+ self.roi_size+self.border_cut, current_y-self.border_cut:current_y + self.roi_size+self.border_cut,
                   current_x-self.border_cut:current_x + self.roi_size + self.border_cut]

        return new_cube

    def __len__(self):
        num_cubes = self.x_steps * self.y_steps * self.z_steps
        return  num_cubes