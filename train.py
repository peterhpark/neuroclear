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
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm

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

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    print ("start the epoch training...")
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1): # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on tensorboard
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                epoch_progress = round(float(epoch_iter/dataset_size),2) * 100
                epoch_count = (epoch-1) * 100  #display the progress in percent: for example 30% past two epochs is 230.
                visualizer.display_current_results(model.get_current_visuals(), epoch_count+epoch_progress)
                visualizer.display_current_histogram(model.get_current_visuals(), epoch)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                print ("----------------------------------")
                print ("exp name: " + str(opt.name) + ", gpu_id:"+str(opt.gpu_ids))
                print ("----------------------------------")
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size

                epoch_progress = round(float(epoch_iter / dataset_size), 2) * 100
                epoch_count = (epoch - 1) * 100  # display the progress in percent: for example 30% past two epochs is 230.
                visualizer.print_current_losses(epoch, epoch_progress, losses, t_comp, t_data)

                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch_count+epoch_progress, losses, is_epoch = False)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                epoch_progress = round(float(epoch_iter / dataset_size), 2) * 100
                epoch_count = (epoch - 1) * 100  # display the progress in percent: for example 30% past two epochs is 230.

                print('saving the latest model (epoch %d, epoch_progress %d%%)' % (epoch, epoch_progress))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
            iter_data_time = time.time()

        # display the image histogram per epoch.
        visualizer.display_current_histogram(model.get_current_visuals(), epoch)

        losses = model.get_current_losses()
        visualizer.plot_current_losses(epoch, losses, is_epoch=True)

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            visualizer.save_current_visuals(model.get_current_visuals(), epoch)

        model.update_learning_rate()
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))