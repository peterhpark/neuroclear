"""
"""

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.assemble_dice import Assemble_Dice
from util import util
from skimage import io
import data
from tqdm import tqdm
import numpy as np
from data.image_folder import make_dataset
from tifffile import imsave

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

    dataset_class = data.find_dataset_using_name(opt.dataset_mode)
    dataset_tolook_shape = dataset_class(opt)
    dataset_size_original = dataset_tolook_shape.size_original()  # return the image size before padding.
    dataset_size = dataset_tolook_shape.size()  # Get the y,x,z volume sizes of the image volume.
    print("original dataset_shape: " + str(dataset_size_original))

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    if opt.data_name == None:
        save_dir = os.path.join(opt.results_dir, opt.name,
                               '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    else:
        save_dir = os.path.join(opt.results_dir, opt.data_name + '_by_' + opt.name,
                               '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
        
    print("Saving results at ... " + str(save_dir))
    if opt.load_iter > 0:  # load_iter is 0 by default
        save_dir = '{:s}_iter{:d}'.format(save_dir, opt.load_iter)

    dice_assembly = Assemble_Dice(opt)  # initialize the dice assembly

    print("whole Image size: {}".format(dice_assembly.image_size))
    print("Whole image step counts y,x,z: {}".format(
        (dice_assembly.y_steps, dice_assembly.x_steps, dice_assembly.z_steps)))
    print("Whole image step counts: {}".format(dice_assembly.y_steps * dice_assembly.x_steps * dice_assembly.z_steps))

    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.

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

    if opt.data_type == 'uint16':
        data_range = 2 ** 16 - 1
        # output_dtype = np.uint16
    elif opt.data_type == 'uint8':
        data_range = 2 ** 8 - 1
        # output_dtype = np.uint8

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

        if opt.load_iter > 0:
            output_xy_vol_path = save_dir + '/volumes/output_volume_xy-view_iter-' + str(opt.load_iter) + '.tif'
        else:
            output_xy_vol_path = save_dir + '/volumes/output_volume_xy-view_epoch-' + str(opt.epoch) + '.tif'
        imsave(output_xy_vol_path, fake_volume)
        print ("Output volume is saved as a tiff file. ")

        if not opt.skip_real:
            input_xy_vol_path = save_dir + '/volumes/input_volume_xy-view.tif'
            imsave(input_xy_vol_path, real_volume)
            print("Input volume is saved as a tiff file. ")


    if opt.dataroot_gt is not None:
        GT_path = make_dataset(opt.dataroot_gt, 1)[0]
        gt_volume = io.imread(GT_path)
        # Ground_truth = Ground_truth[-z:, -y:, -x:] #crop to match the cropped input and output

        print("Calculating PSNR for the whole image volume...")

        ##
        # Calculate image metrics

        datarange = 2**8-1

        real_volume = util.normalize(util.standardize(real_volume), data_type=np.uint8)
        fake_volume = util.normalize(util.standardize(fake_volume), data_type=np.uint8)
        gt_volume = util.normalize(util.standardize(gt_volume), data_type=np.uint8)

        real_volume = util.normalize(util.standardize(real_volume), data_type=np.uint8)
        fake_volume = util.normalize(util.standardize(fake_volume), data_type=np.uint8)
        gt_volume = util.normalize(util.standardize(gt_volume), data_type=np.uint8)

        psnr_input_gt = util.get_psnr(real_volume, gt_volume, datarange)
        psnr_output_gt = util.get_psnr(fake_volume, gt_volume, datarange)
        print ("Metrics are calculated.")

        message = 'Experiment Name: ' + opt.name + '\n'
        message += '---------------------------------------------------------\n'
        message += '\nWhole_volume\n'
        message += '---------------------------------------------------------\n'
        message += 'Network Input vs. Groundtruth\n'
        message += '(psnr: %.4f) \n' % (
        psnr_input_gt)
        message += '---------------------------------------------------------\n'
        message += 'Network Output vs. Groundtruth\n'
        message += '(psnr: %.4f) \n' % (
        psnr_output_gt)
        message += '---------------------------------------------------------'

        print (message)
        filename = os.path.join(save_dir, 'metrics.txt')

        with open(filename, "a") as metric_file:
            metric_file.write('%s\n' % message)  # save the message
    print("----Test done----")