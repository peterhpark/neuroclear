"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import html
from util.assemble_dice import Assemble_Dice
from util import util
from skimage import io
import data
from tqdm import tqdm
import numpy as np
from skimage.metrics import structural_similarity as get_ssim
from skimage.metrics import peak_signal_noise_ratio as get_psnr
from skimage.metrics import normalized_root_mse as get_nrmse
from skimage.metrics import mean_squared_error as get_mse

from data.image_folder import make_dataset
from tifffile import imsave

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    ## DEBUG FLAG
    if opt.debug:
        print("DEBUG MODE ACTIVATED.")
        import pydevd_pycharm

        Host_IP_address = '143.248.31.79'
        print("For debug, listening to...{}".format(Host_IP_address))
        # pydevd_pycharm.settrace('143.248.31.79', port=5678, stdoutToServer=True, stderrToServer=True)
        pydevd_pycharm.settrace(Host_IP_address, port=5678, stdoutToServer=True, stderrToServer=True)
    ##

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

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name,
                           '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory

    print("web_dir: " + str(web_dir))
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)

    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

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
    #dice_assembly.assemble_all(imtype = output_type, background_threshold=( bckgr_thre, bckgr_val))
    print("Image volume re-assembled.")
    img_whole_dict = dice_assembly.getDict()
    print("re-merged image shape: {}".format(img_whole_dict['real'].shape))
    webpage_wholeimg = html.HTML(web_dir, 'Whole_img: Experiment = %s, Phase = %s, Epoch = %s' % (
    opt.name, opt.phase, opt.epoch))

    # print ("Saving the whole image volume...")
    #
    # util.mkdir(web_dir + '/images/output')
    # output_path = web_dir + '/images/output/output.tif'
    #
    # util.mkdir(web_dir + '/images/input')
    # input_path = web_dir + '/images/input/input.tif'

    if opt.data_type == 'uint16':
        data_range = 2 ** 16 - 1
        # output_dtype = np.uint16
    elif opt.data_type == 'uint8':
        data_range = 2 ** 8 - 1
        # output_dtype = np.uint8

    ############# Change the image type ##################
    real_volume = img_whole_dict['real']
    # real_volume = real_volume.astype(opt.data_type)
    fake_volume = img_whole_dict['fake']
    # fake_volume = fake_volume.astype(opt.data_type)
    print ("Data type is: " + str(fake_volume.dtype))
    z, y, x = real_volume.shape
    #####################################################

    # imsave(input_path, real_volume)
    # imsave(output_path, fake_volume)
    #
    # print ("Whole image volumes are saved. ")

    if opt.dataroot_gt is not None:
        from skimage.exposure import match_histograms

        util.mkdir(web_dir + '/images/gt')
        GT_save_path = web_dir + '/images/gt/GT.tif'
        GT_path = make_dataset(opt.dataroot_gt, 1)[0]
        Ground_truth = io.imread(GT_path)
        Ground_truth = Ground_truth[-z:, -y:, -x:] #crop to match the cropped input and output
        if opt.data_type == 'uint16' and Ground_truth.dtype == np.uint8:
            print ("GT is 8-bit.")
            Ground_truth = Ground_truth / (2**8-1)
            Ground_truth *= (2**16-1)
            Ground_truth = Ground_truth.astype(np.uint16)

        bckgr_thre, bckgr_val = opt.background_threshold
        slice_start, slice_end = opt.reference_slice_range

        print("Ground-truth image loaded.")


        # Match input and output histogram to GT.
        # fake_volume = match_histograms(fake_volume, Ground_truth)
        # real_volume = match_histograms(real_volume, Ground_truth)
        # print ("applied histogram matching on both fake and real to GT.")


        print("Calculating SSIM and PSNR for the whole image volume...")

        ##
        # Calculate image metrics
        # dims are [input:0/output:1, mean metric across XY slice, mean metric across XZ slice, mean metric across YZ slice

        vol_shape = real_volume.shape
        ssim_array = np.zeros((2, 3)) # SSIM
        psnr_array = np.zeros((2, 3)) # PSNR
        nrmse_array = np.zeros((2, 3)) # Normalized RMSE
        mse_array = np.zeros((2, 3)) # MSE

        ## Calculate for XY
        ssim_input_list_xy = []
        psnr_input_list_xy = []
        nrmse_input_list_xy = []
        mse_input_list_xy = []

        ssim_output_list_xy = []
        psnr_output_list_xy = []
        nrmse_output_list_xy = []
        mse_output_list_xy = []

        for sl_index, xy_slice_input in enumerate(real_volume):
            ssim_input_list_xy.append(get_ssim(Ground_truth[sl_index], xy_slice_input))
            psnr_input_list_xy.append(get_psnr(Ground_truth[sl_index], xy_slice_input, data_range=data_range))
            nrmse_input_list_xy.append(get_nrmse(Ground_truth[sl_index], xy_slice_input))
            mse_input_list_xy.append(get_mse(Ground_truth[sl_index], xy_slice_input))

        for sl_index, xy_slice_output in enumerate(fake_volume):
            ssim_output_list_xy.append(get_ssim(Ground_truth[sl_index], xy_slice_output))
            psnr_output_list_xy.append(get_psnr(Ground_truth[sl_index], xy_slice_output, data_range=data_range))
            nrmse_output_list_xy.append(get_nrmse(Ground_truth[sl_index], xy_slice_output))
            mse_output_list_xy.append(get_mse(Ground_truth[sl_index], xy_slice_output))

        ssim_array[0, 0] = np.median(ssim_input_list_xy)
        ssim_array[1, 0] = np.median(ssim_output_list_xy)

        psnr_array[0, 0] = np.median(psnr_input_list_xy)
        psnr_array[1, 0] = np.median(psnr_output_list_xy)

        nrmse_array[0, 0] = np.median(nrmse_input_list_xy)
        nrmse_array[1, 0] = np.median(nrmse_output_list_xy)

        ## Calculate for XZ
        ssim_input_list_xz = []
        psnr_input_list_xz = []
        nrmse_input_list_xz = []
        mse_input_list_xz = []

        ssim_output_list_xz = []
        psnr_output_list_xz = []
        nrmse_output_list_xz = []
        mse_output_list_xz = []

        for sl_index in range(vol_shape[1]):
            ssim_input_list_xz.append(get_ssim(Ground_truth[:, sl_index], real_volume[:,sl_index]))
            psnr_input_list_xz.append(get_psnr(Ground_truth[:, sl_index], real_volume[:,sl_index], data_range=data_range))
            nrmse_input_list_xz.append(get_nrmse(Ground_truth[:, sl_index], real_volume[:,sl_index]))
            mse_input_list_xz.append(get_mse(Ground_truth[:, sl_index], real_volume[:,sl_index]))

        for sl_index in range(vol_shape[1]):
            ssim_output_list_xz.append(get_ssim(Ground_truth[:, sl_index], fake_volume[:,sl_index]))
            psnr_output_list_xz.append(get_psnr(Ground_truth[:, sl_index], fake_volume[:,sl_index], data_range=data_range))
            nrmse_output_list_xz.append(get_nrmse(Ground_truth[:, sl_index], fake_volume[:,sl_index]))
            mse_output_list_xz.append(get_mse(Ground_truth[:, sl_index], fake_volume[:,sl_index]))

        ssim_array[0, 1] = np.median(ssim_input_list_xz)
        ssim_array[1, 1] = np.median(ssim_output_list_xz)

        psnr_array[0, 1] = np.median(psnr_input_list_xz)
        psnr_array[1, 1] = np.median(psnr_output_list_xz)

        nrmse_array[0, 1] = np.median(nrmse_input_list_xz)
        nrmse_array[1, 1] = np.median(nrmse_output_list_xz)

        ssim_input_list_yz = []
        psnr_input_list_yz = []
        nrmse_input_list_yz = []
        mse_input_list_yz = []

        ssim_output_list_yz = []
        psnr_output_list_yz = []
        nrmse_output_list_yz = []
        mse_output_list_yz = []

        for sl_index in range(vol_shape[2]):
            ssim_input_list_yz.append(get_ssim(Ground_truth[:, :, sl_index], real_volume[:, :, sl_index],
                                      data_range=data_range))
            psnr_input_list_yz.append(get_psnr(Ground_truth[:, :, sl_index], real_volume[:, :, sl_index],
                                      data_range=data_range))
            nrmse_input_list_yz.append(get_nrmse(Ground_truth[:, :, sl_index], real_volume[:, :, sl_index]))
            mse_input_list_yz.append(get_mse(Ground_truth[:, :, sl_index], real_volume[:, :, sl_index]))

        for sl_index in range(vol_shape[2]):
            ssim_output_list_yz.append(get_ssim(Ground_truth[:, :, sl_index], fake_volume[:, :, sl_index],
                                       data_range=data_range))
            psnr_output_list_yz.append(get_psnr(Ground_truth[:, :, sl_index], fake_volume[:, :, sl_index],
                                       data_range=data_range))
            nrmse_output_list_yz.append(get_nrmse(Ground_truth[:, :, sl_index], fake_volume[:, :, sl_index]))
            mse_output_list_yz.append(get_mse(Ground_truth[:, :, sl_index], fake_volume[:, :, sl_index]))

        ssim_array[0, 2] = np.median(ssim_input_list_yz)
        ssim_array[1, 2] = np.median(ssim_output_list_yz)

        psnr_array[0, 2] = np.median(psnr_input_list_yz)
        psnr_array[1, 2] = np.median(psnr_output_list_yz)

        nrmse_array[0, 2] = np.median(nrmse_input_list_yz)
        nrmse_array[1, 2] = np.median(nrmse_output_list_yz)

        print("calculating the metrics for whole image volumes")
        # ssim_input_gt = get_ssim(Ground_truth, real_volume, data_range = data_range)
        ssim_input_gt = get_ssim(Ground_truth, real_volume)
        psnr_input_gt = get_psnr(Ground_truth, real_volume, data_range = data_range)
        mse_input_gt = get_mse(Ground_truth, real_volume)
        nrmse_input_gt = get_nrmse(Ground_truth, real_volume)

        # ssim_output_gt = get_ssim(Ground_truth, fake_volume, data_range = data_range)
        ssim_output_gt = get_ssim(Ground_truth, fake_volume)
        psnr_output_gt = get_psnr(Ground_truth, fake_volume, data_range = data_range)
        mse_output_gt = get_mse(Ground_truth, fake_volume)
        nrmse_output_gt = get_nrmse(Ground_truth, fake_volume)
        #

        print ("Metrics are calculated.")

        message = 'Experiment Name: ' + opt.name + '\n'
        message += '---------------------------------------------------------\n'
        message += '\nIn XY plane\n'
        message += '---------------------------------------------------------\n'

        message += 'Network Input vs. Groundtruth\n'
        message += '(ssim: %.4f, psnr: %.4f, mse: %.4f, nrmse: %.4f) \n' % (
        ssim_array[0,0], psnr_array[0,0], mse_array[0,0], nrmse_array[0,0])
        message += '---------------------------------------------------------\n'
        message += 'Network Output vs. Groundtruth\n'
        message += '(ssim: %.4f, psnr: %.4f, mse: %.4f, nrmse: %.4f) \n' % (
        ssim_array[1,0], psnr_array[1,0], mse_array[1,0], nrmse_array[1,0])
        message += '---------------------------------------------------------'

        message += '\nIn XZ plane\n'
        message += '---------------------------------------------------------\n'
        message += 'Network Input vs. Groundtruth\n'
        message += '(ssim: %.4f, psnr: %.4f, mse: %.4f, nrmse: %.4f) \n' % (
        ssim_array[0,1], psnr_array[0,1], mse_array[0,1], nrmse_array[0,1])
        message += '---------------------------------------------------------\n'
        message += 'Network Output vs. Groundtruth\n'
        message += '(ssim: %.4f, psnr: %.4f, mse: %.4f, nrmse: %.4f) \n' % (
        ssim_array[1,1], psnr_array[1,1], mse_array[1,1], nrmse_array[1,1])
        message += '---------------------------------------------------------'

        message += '\nIn YZ plane\n'
        message += '---------------------------------------------------------\n'
        message += 'Network Input vs. Groundtruth\n'
        message += '(ssim: %.4f, psnr: %.4f, mse: %.4f, nrmse: %.4f) \n' % (
        ssim_array[0,2], psnr_array[0,2], mse_array[0,2], nrmse_array[0,2])
        message += '---------------------------------------------------------\n'
        message += 'Network Output vs. Groundtruth\n'
        message += '(ssim: %.4f, psnr: %.4f, mse: %.4f, nrmse: %.4f) \n' % (
        ssim_array[1,2], psnr_array[1,2], mse_array[1,2], nrmse_array[1,2])
        message += '---------------------------------------------------------'

        # message += '---------------------------------------------------------\n'
        # message += '\nWhole_volume\n'
        # message += '---------------------------------------------------------\n'
        # message += 'Network Input vs. Groundtruth\n'
        # message += '(ssim: %.4f, psnr: %.4f, mse: %.4f, nrmse: %.4f) \n' % (
        # ssim_input_gt, psnr_input_gt, mse_input_gt, nrmse_input_gt)
        # message += '---------------------------------------------------------\n'
        # message += 'Network Output vs. Groundtruth\n'
        # message += '(ssim: %.4f, psnr: %.4f, mse: %.4f, nrmse: %.4f) \n' % (
        # ssim_output_gt, psnr_output_gt, mse_output_gt, nrmse_output_gt)
        # message += '---------------------------------------------------------'

        print (message)
        filename = os.path.join(web_dir, 'metrics.txt')

        with open(filename, "a") as metric_file:
            metric_file.write('%s\n' % message)  # save the message

    if opt.save_volume:
        util.mkdir(web_dir + '/volumes')
        # output_xy_vol_path = web_dir + '/volumes/output_volume_xy-view.tif'

        if opt.load_iter > 0:
            output_xy_vol_path = web_dir + '/volumes/output_volume_xy-view_iter-' + str(opt.load_iter) + '.tif'
        else:
            output_xy_vol_path = web_dir + '/volumes/output_volume_xy-view_epoch-' + str(opt.epoch) + '.tif'
        imsave(output_xy_vol_path, fake_volume)
        print ("Output volume is saved as a tiff file. ")

        if not opt.skip_real:
            input_xy_vol_path = web_dir + '/volumes/input_volume_xy-view.tif'
            imsave(input_xy_vol_path, real_volume)
            print("Input volume is saved as a tiff file. ")

    # if opt.save_projection:
    #     start_slice, end_slice = opt.projection_range
    #     fake_proj_xy = np.amax(fake_volume[start_slice:end_slice], axis=0)
    #     fake_proj_xz = np.amax(fake_volume[start_slice:end_slice], axis=1)
    #     fake_proj_yz = np.amax(fake_volume, axis=2)
    #
    #     util.mkdir(web_dir + '/projections')
    #
    #     util.save_image(fake_proj_xy, web_dir + '/projections/fake_xy_proj.tif')
    #     util.save_image(fake_proj_xz, web_dir + '/projections/fake_xz_proj.tif')
    #     util.save_image(fake_proj_yz, web_dir + '/projections/fake_yz_proj.tif')
    #
    #     if not opt.skip_real:
    #         real_proj_xy = np.amax(real_volume, axis=0)
    #         real_proj_xz = np.amax(real_volume, axis=1)
    #         real_proj_yz = np.amax(real_volume, axis=2)
    #
    #         util.save_image(real_proj_xy, web_dir + '/projections/real_xy_proj.tif')
    #         util.save_image(real_proj_xz, web_dir + '/projections/real_xz_proj.tif')
    #         util.save_image(real_proj_yz, web_dir + '/projections/real_yz_proj.tif')

    if opt.save_slices:
        output_xy_path = web_dir + '/images/output_xy/output_xy_'
        output_yz_path = web_dir + '/images/output_yz/output_yz_'
        output_xz_path = web_dir + '/images/output_xz/output_xz_'

        util.mkdir(web_dir + '/images/output_xy')
        util.mkdir(web_dir + '/images/output_yz')
        util.mkdir(web_dir + '/images/output_xz')

        if not opt.skip_real:
            input_xy_path = web_dir + '/images/input_xy/input_xy_'
            input_yz_path = web_dir + '/images/input_yz/input_yz_'
            input_xz_path = web_dir + '/images/input_xz/input_xz_'

            util.mkdir(web_dir + '/images/input_xy')
            util.mkdir(web_dir + '/images/input_yz')
            util.mkdir(web_dir + '/images/input_xz')

        if opt.dataroot_gt is not None:
            gt_xy_path = web_dir + '/images/gt_xy/gt_xy_'
            gt_yz_path = web_dir + '/images/gt_yz/gt_yz_'
            gt_xz_path = web_dir + '/images/gt_xz/gt_xz_'

            util.mkdir(web_dir + '/images/gt_xy')
            util.mkdir(web_dir + '/images/gt_yz')
            util.mkdir(web_dir + '/images/gt_xz')

        for i in tqdm(range(img_whole_dict['real'].shape[2])):
            # Save yz slices

            util.save_image(fake_volume[:,:,i], output_yz_path + str(i) + '.tif')

            if not opt.skip_real:
                util.save_image(real_volume[:, :, i], input_yz_path + str(i) + '.tif')

            if opt.dataroot_gt is not None:
                util.save_image(Ground_truth[:, :, i], gt_yz_path + str(i) + '.tif')

        for i in range(img_whole_dict['real'].shape[1]):
            # Save xz slices

            util.save_image(fake_volume[:,i,:], output_xz_path + str(i) + '.tif')
            if not opt.skip_real:
                util.save_image(real_volume[:,i,:], input_xz_path + str(i) + '.tif')

            if opt.dataroot_gt is not None:
                util.save_image(Ground_truth[:, i, :], gt_xz_path + str(i) + '.tif')

        for i in tqdm(range(img_whole_dict['real'].shape[0])):
            # Save xy slices
            snapshot_xy = dice_assembly.getSnapshots(i, slice_axis=0)

            util.save_image(fake_volume[i,:,:], output_xy_path + str(i) + '.tif')
            if not opt.skip_real:
                util.save_image(real_volume[i,:,:], input_xy_path + str(i) + '.tif')

            if opt.dataroot_gt is not None:
                util.save_image(Ground_truth[i,:,:], gt_xy_path + str(i) + '.tif')


    print("----Test done----")