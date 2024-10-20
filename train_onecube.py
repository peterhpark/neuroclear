"""
"""
import time
from options.base_options import BaseOptions
import data
from models import create_model
from util.visualizer import Visualizer


if __name__ == '__main__':
    print ("Training session started.")
    opt = BaseOptions().gather_options() # load configs from an YAML file 
    
    dataset_class = data.find_dataset_using_name(opt.dataset_mode) # type: ignore
    dataset = dataset_class(opt)

    n_epochs = opt.n_epochs # type: ignore # how many epochs to run?
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    visualizer.define_metrics(model.loss_names) # set loss metrics 
    iter_data_time = time.time()    # timer for data loading per iteration
    total_iters = 0  # the number of training iterations in current epoch, reset to 0 every epoch

    if opt.load_iter > 0:
        loaded_iter = opt.load_iter+1
    else:
        loaded_iter = 0

    total_iters = total_iters + loaded_iter

    dataset_len = len(dataset)
    epoch = 0

    while True: #FIXME: with our current dataset loader, it NEVER ends out of a loop, because of Pytorch's internal index counting
        for i, data in enumerate(dataset): 
            iter_start_time = time.time()  # timer for computation per iteration
            # if (total_iters-loaded_iter) % opt.print_freq == 0:
            #     t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on wandb
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), total_iters, commit=False)
                losses = model.get_current_losses()
                visualizer.plot_current_losses(losses, total_iters)

            if total_iters % dataset_len//2 == 0:   # cache our latest model every <save_latest_freq> iterations
            # if i % 3 == 0:   # cache our latest model every <save_latest_freq> iterations
                save_suffix = 'iter_%d' % total_iters
                model.save_networks(save_suffix)
                iter_data_time = time.time()
                # model.update_learning_rate()  # update here at the end of every epoch
                epoch += 1
                print (f"End of Epoch #{epoch}")
