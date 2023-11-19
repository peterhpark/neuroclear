"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md

### Train script that follows the original cycleGAN training routine, with no repetition.###

"""
import time
from options.train_options import TrainOptions
import data
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm

import numpy as np

#TODO this doesn't work. 

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    ## DEBUG FLAG
    if opt.debug:
        print ("DEBUG MODE ACTIVATED.")
        import pydevd_pycharm
        Host_IP_address = '143.248.31.79'
        print ("For debug, listening to...{}".format(Host_IP_address))
        # pydevd_pycharm.settrace('143.248.31.79', port=5678, stdoutToServer=True, stderrToServer=True)
        pydevd_pycharm.settrace(Host_IP_address, port=5678, stdoutToServer=True, stderrToServer=True)
    ##

    dataset_class = data.find_dataset_using_name(opt.dataset_mode)
    dataset = dataset_class(opt)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots

    iter_data_time = time.time()    # timer for data loading per iteration
    total_iters = 0  # the number of training iterations in current epoch, reset to 0 every epoch

    if opt.load_iter > 0:
        loaded_iter = opt.load_iter+1
    else:
        loaded_iter = 0

    total_iters = total_iters + loaded_iter

    visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
    visualizer.display_model_hyperparameters()

    #TODO Move print statements to a txt file.
    while True: # infinite training within epoch

        random_index = np.random.randint(0,10)
        data = dataset[random_index]

        iter_start_time = time.time()  # timer for computation per iteration

        if (total_iters-loaded_iter) % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        total_iters += opt.batch_size
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

        if total_iters % opt.display_freq == 0:   # display images on tensorboard
            save_result = total_iters % opt.update_html_freq == 0
            model.compute_visuals()
            visualizer.display_current_results(model.get_current_visuals(), total_iters)

        if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            print ("----------------------------------")
            print ("exp name: " + str(opt.name) + ", gpu_id:"+str(opt.gpu_ids))
            print ("----------------------------------")
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            # visualizer.print_current_losses(1, total_iters, losses, t_comp, t_data)

            if opt.display_id > 0:
                visualizer.plot_current_losses(total_iters, losses, is_epoch = False)

        if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            print ("----------------------------------")
            print('saving the latest model (iteration %d)' % total_iters)
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            model.save_networks(save_suffix)
            print('saving the current histogram (iteration %d)' % total_iters)
            visualizer.display_current_histogram(model.get_current_visuals(), total_iters)
            # print('saving the current visuals (iteration %d)' % total_iters)
            # visualizer.save_current_visuals(model.get_current_visuals(), total_iters)


        model.update_learning_rate()  # update here instead of at the end of every epoch
        iter_data_time = time.time()
