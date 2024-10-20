"""
"""

import os
from data import create_dataset
from models import create_model
from options.base_options import BaseOptions
from util.assemble_dice import Assemble_Dice
from util import util
from skimage import io
import data
from tqdm import tqdm
import numpy as np
# from skimage.metrics import structural_similarity as get_ssim
# from skimage.metrics import peak_signal_noise_ratio as get_psnr
# from skimage.metrics import normalized_root_mse as get_nrmse
# from skimage.metrics import mean_squared_error as get_mse

from data.image_folder import make_dataset
from tifffile import imsave

if __name__ == '__main__':
    opt = BaseOptions().gather_options() # load configs from an YAML file 

    opt.dataset_mode = 'diceImage' 

    # hard-code some parameters for test
    opt.num_threads = 1  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.

    # not necessary
    dataset_class = data.find_dataset_using_name(opt.dataset_mode)
    dataset_tolook_shape = dataset_class(opt)
    dataset_size_original = dataset_tolook_shape.size_original()  # return the image size before padding.
    dataset_size = dataset_tolook_shape.size()  # Get the y,x,z volume sizes of the image volume.
    print("original dataset_shape: " + str(dataset_size_original))

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other option
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    save_dir = os.path.join(opt.results_dir, opt.name,
                            '{}_{}'.format(opt.phase, opt.load_iter))  # define the website directory



    if opt.load_iter > 0:  # load_iter is 0 by default
        save_dir = '{:s}_iter{:d}'.format(save_dir, opt.load_iter)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    print("Saving results at ... " + str(save_dir))

    dice_assembly = Assemble_Dice(opt)  # initialize the dice assembly

    print("whole Image size: {}".format(dice_assembly.image_size))
    print("Whole image step counts y,x,z: {}".format(
        (dice_assembly.y_steps, dice_assembly.x_steps, dice_assembly.z_steps)))
    print("Whole image step counts: {}".format(dice_assembly.y_steps * dice_assembly.x_steps * dice_assembly.z_steps))

    if opt.eval:
        model.eval()

    for i, data in enumerate(tqdm(dataset)):
        # data dimension: batch, color_channel, z, y, x
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        dice_assembly.addToStack(visuals)  # converts tensor to img and add to stack.

    print("Inference Done. ")

    dice_assembly.assemble_all()
    print("Image volume re-assembled.")
    img_whole_dict = dice_assembly.getDict()
    print("re-merged image shape: {}".format(img_whole_dict['fake'].shape))


    ############# Change the image type ##################
    if not opt.skip_real:
        real_volume = img_whole_dict['real']
        # real_volume = real_volume.astype(opt.data_type)
        print ("Input data type is: " + str(real_volume.dtype))

    fake_volume = img_whole_dict['fake']
    # fake_volume = fake_volume.astype(opt.data_type)
    print ("Output data type is: " + str(fake_volume.dtype))
    #####################################################

    if opt.save_volume:
        util.mkdir(save_dir + '/volumes')
        output_xy_vol_path = save_dir + '/volumes/output_volume_xy-view_iter-' + str(opt.load_iter) + '.tif'
        imsave(output_xy_vol_path, fake_volume)
        print ("Output volume is saved as a tiff file. ")

        if not opt.skip_real:
            input_xy_vol_path = save_dir + '/volumes/input_volume_xy-view.tif'
            imsave(input_xy_vol_path, real_volume)
            print("Input volume is saved as a tiff file. ")

    print("----Test done----")