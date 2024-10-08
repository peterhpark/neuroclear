"""
"""
import time
from options.train_options import TrainOptions
from options.base_options import BaseOptions
import data
from models import create_model
from util.visualizer import Visualizer


if __name__ == '__main__':
    print ("TEST")
    opt = BaseOptions().gather_options() # load configs from an YAML file 
    
    dataset_class = data.find_dataset_using_name(opt.dataset_mode) # type: ignore
    dataset = dataset_class(opt)

    n_epochs = opt.n_epochs # type: ignore # how many epochs to run?
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

    for epoch in range(n_epochs):
        for i, data in enumerate(dataset):
            iter_start_time = time.time()  # timer for computation per iteration
            if (total_iters-loaded_iter) % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on tensorboard
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), total_iters)
                visualizer.plot_current_losses(total_iters, losses, is_epoch = False)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
 
            model.update_learning_rate()  # update here instead of at the end of every epoch

        if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            model.save_networks(save_suffix)

            iter_data_time = time.time()
