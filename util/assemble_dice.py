# TODO Sep 08 version
import numpy as np
from collections import OrderedDict
import scipy.signal
from util import util
from skimage.exposure import match_histograms
import data
import torch

class Assemble_Dice():
    def __init__(self, opt):
        dataset_class = data.find_dataset_using_name(opt.dataset_mode)
        dataset_tolook_shape = dataset_class(opt)
        self.image_size_original = dataset_tolook_shape.size_original()  # return the image size before padding.
        self.image_size = dataset_tolook_shape.size() # Get the y,x,z volume sizes of the image volume.
        self.border_cut = opt.border_cut

        self.roi_size = opt.dice_size[0]
        self.overlap = opt.overlap
        self.step = self.roi_size - self.overlap

        self.z_steps  = (self.image_size[0]-self.overlap)//self.step
        self.y_steps = (self.image_size[1]-self.overlap)//self.step
        self.x_steps  = (self.image_size[2]-self.overlap)//self.step

        self.visual_ret = OrderedDict()
        self.visual_names = ['real', 'fake']
        self.snapDict = OrderedDict()
        self.cube_queue = OrderedDict()
        self.mask_ret = OrderedDict()
        self.imtype = opt.data_type

        self.no_histogram_match = opt.no_histogram_match
        if not(self.no_histogram_match):
            print ("We will match the histograms of output sub-volumes with input sub-volumes.")

        if ('normalizedcgan' in opt.preprocess):
            print ("Converting the image type based on DC-GAN normalization range...")
            self.use_dcgan_norm = True
        else:
            self.use_dcgan_norm = False

        self.len_cube_queue = self.z_steps  * self.x_steps * self.y_steps # total number of cubes

        # initialize the mapping.
        for name in self.visual_names:
            self.visual_ret[name] =np.zeros(self.image_size, dtype = np.float32)
            self.mask_ret[name] = np.zeros(self.image_size,  dtype = np.float32)
            self.cube_queue[name] = []

    def indexTo3DIndex(self, index):
        # Dicing order: x-> y-> z
        x_cube_index = index % self.x_steps
        y_cube_index = (index % (self.x_steps*self.y_steps))//self.x_steps
        z_cube_index = (index) // (self.x_steps*self.y_steps)

        return z_cube_index, y_cube_index, x_cube_index

    def indexToCoordinates(self, index): # converts 1D dicing order to 3D stacking order

        # Dicing order: x-> y-> z
        z_cube_index, y_cube_index, x_cube_index = self.indexTo3DIndex(index)

        current_z = z_cube_index * (self.step)
        current_y = y_cube_index * (self.step)
        current_x = x_cube_index * (self.step)

        return current_z, current_y, current_x

    def varycubeinput(self, input):
        # takes visual dictionary and creates a list of augmented copies of the visual.
        data_name = list(input.keys())# TODO: there's a difference between the data names in input and output: A vs. B <-> real vs. fake
        input_visual = input[data_name[0]]
        input_path = input[data_name[1]]
        axis_len = len(input_visual.shape)
        axes = np.arange(2, axis_len)

        input_list = []
        input_list.append(input)

        for axis in axes: # per axis of flipping, add each augmented copy to a list
            input_dict = OrderedDict()
            axis = int(axis)
            input_dict[data_name[0]] = input_visual.flip(axis)
            input_dict[data_name[1]] = input_path
            input_list.append(input_dict)
        # In dict_list, every item is an augmented copy by each augmentation parameter.

        return input_list

    def combinecube(self, visual_list):
        visual_dict_sample = visual_list[0]
        keys = list(visual_dict_sample.keys())
        axis_len = len(visual_dict_sample[keys[0]].shape)
        axes = np.arange(2, axis_len)

        dict_list = []
        dict_list.append(visual_list[0]) # include the unchanged.
        visual_list.pop(0) # remove the unchanged from the list

        # dimensions are: cube_variation, batch_num, color_channel, z, y, x
        for i, flip_var in enumerate(visual_list):
            visual_dict = OrderedDict()
            for name in keys:
                axis = int(axes[i])
                visual_dict[name] = flip_var[name].flip(axis) # unflip them
            dict_list.append(visual_dict)

        visual_recon_dict = OrderedDict()

        for name in keys:
            cube_list = []
            for i, unflip_var in enumerate(dict_list):
                cube_list.append(unflip_var[name])

            cube_stack = torch.stack(cube_list, dim=0)
            visual_recon_dict[name] = torch.mean(cube_stack, dim=0)

        return visual_recon_dict

    def addToStack(self, cube): # Cube is an orderedDict with visual_names as keys.
        cube_dict = OrderedDict()
        for name in self.visual_names:
            image_tensor = cube[name]
            image_numpy_og = image_tensor.cpu().float().numpy()  # convert it into a numpy array
            cube_numpy = image_numpy_og.copy()

            _, _, h, w, d = cube_numpy.shape # When we add an output cube from the network, it include two extra dimensions: batch_index, color_channel
            cube_numpy = cube_numpy.squeeze() # remove the batch and color channel axis.

            cube_numpy = cube_numpy.astype(np.float32) # saves memory

            # Remove the border regions to avoid the popping effect.
            cube_numpy = cube_numpy[self.border_cut:-self.border_cut, self.border_cut:-self.border_cut, self.border_cut:-self.border_cut]

            assert cube_numpy.shape == (self.roi_size, self.roi_size, self.roi_size), "the cube dimensions are invalid."

            cube_dict[name] = cube_numpy

        ## Perform histogram matching on the output cube to the input cube as post-processing.
        if not (self.no_histogram_match):
            cube_dict['fake'] = match_histograms(cube_dict['fake'], cube_dict['real'])
        ##

        for name in self.visual_names:
            self.cube_queue[name].append(cube_dict[name])

    def assemble_all(self):
        for name in self.visual_names:
            print ("Patching for... " + str(name))
            for index, cube in enumerate(self.cube_queue[name]):
                current_z, current_y, current_x = self.indexToCoordinates(index)

                # assert cube.dtype == self.imtype, "Data type of the assembling cubes does not match the given data type. "
                if self.overlap > 0:
                    self.visual_ret[name][current_z:current_z + self.roi_size, current_y:current_y + self.roi_size,
                    current_x:current_x + self.roi_size] += cube/8 # divide by 4 to prevent the overflowing.
                    self.mask_ret[name][current_z:current_z + self.roi_size, current_y:current_y + self.roi_size,
                    current_x:current_x + self.roi_size] += np.ones((self.roi_size,self.roi_size,self.roi_size),  dtype = np.float32)
                if cube.shape != (self.roi_size, self.roi_size, self.roi_size):
                    raise Exception('The cube does not have the proper size.')

            print ("done patching the cubes for {} image volume.".format(str(name)))

            if self.overlap > 0:
                print("merging all gaps by linear averaging for {} image volume...".format(str(name)))

                self.visual_ret[name] = (self.visual_ret[name]/self.mask_ret[name])*8  # multiply by 4 to recover the original values without overflowing from earlier.
                print("All gaps merged for {} image volume.".format(str(name)))

            print ("For debug: maximum iterations of overlaps: " + str(np.max(self.mask_ret[name])))

        ## convert the datatype
        if self.imtype == 'uint8':
            # self.visual_ret[name] *= self.img_std
            # self.visual_ret[name] += self.img_mean # then the image is scaled to 0-1.

            self.visual_ret[name] *= 255
            self.visual_ret[name] = self.visual_ret[name].astype(np.uint8)

        if self.imtype == 'uint16':
            # self.visual_ret[name] *= self.img_std
            # self.visual_ret[name] += self.img_mean # then the image is scaled to 0-1.

            self.visual_ret[name] *= 2 ** 16 - 1
            self.visual_ret[name] = self.visual_ret[name].astype(np.uint16)

        # crop the regions that were padded for clean-cut dicing.
        if self.image_size_original is not None:
            padders_ = [self.image_size[i] - self.image_size_original[i] for i in range(len(self.image_size))]
            print("Image cropped to revert back to the original size by: " + str(padders_))
            self.visual_ret[name] = self.visual_ret[name][:-padders_[0], :-padders_[1], :-padders_[2]]


    # tells you if the index corresponds to a cube outside the boundary of the whole image.
    def if_overEdge(self, index):
        z_cube_index, y_cube_index, x_cube_index  = self.indexTo3DIndex(index)

        z_over = z_cube_index > self.z_steps or z_cube_index < 0
        y_over = y_cube_index > self.y_steps or y_cube_index < 0
        x_over = x_cube_index > self.x_steps or x_cube_index < 0
        all_over = index >  (self.len_cube_queue-1)

        return z_over or y_over or x_over or all_over

    # slice image across z-depth
    def getSnapshots(self, index, slice_axis=2):
        for name in self.visual_names:
            if slice_axis ==0:
                self.snapDict[name] = self.visual_ret[name][index, :, :]
            if slice_axis ==1:
                self.snapDict[name] = self.visual_ret[name][:, index, :]
            if slice_axis ==2:
                self.snapDict[name] = self.visual_ret[name][:,:,index]
        return self.snapDict

    def getDict(self):
        return self.visual_ret

    def getMaskRet(self):
        return self.mask_ret['real']

    def getCubeQueue(self):
        return self.cube_queue
