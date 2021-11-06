import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import html
from assemble_dice import Assemble_Dice
from testsuite import dummy_opt
from util import util
import numpy as np
import data


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    dataset_class = data.find_dataset_using_name(opt.dataset_mode)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    dataset_class_2 = data.find_dataset_using_name(opt.dataset_mode)
    dummy_opt = dummy_opt.Dummpy_Opt(dataroot = opt.dataroot_gt, crop_size=opt.crop_size, preprocess=opt.preprocess)
    dataset_ref = dataset_class_2(dummy_opt)

    # create a website
    if opt.data_name == None:
        web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    else:
        web_dir = os.path.join(opt.results_dir, opt.data_name + '_by_' + opt.name,
                               '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print ("web_dir: " + str(web_dir))
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)

    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    for i, data in enumerate(dataset):

        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results

        real_img = util.tensor2im(visuals['real'], imtype=np.uint16, is_normalize=True).squeeze()
        fake_img = util.tensor2im(visuals['fake'], imtype=np.uint16, is_normalize=True).squeeze()

        if psnr_output == float('Inf'):
            psnr_output = 100

from tifffile import imsave

message = 'Experiment Name: ' + opt.name + '\n'
message += '---------------------------------------------------------\n'
message += 'Network Input vs. Groundtruth\n'
message += '(ssim: %.4f, psn: %.4f \n) ' % (ssim_input, psnr_input)
message += '---------------------------------------------------------\n'
message += 'Network Output vs. Groundtruth\n'
message += '(ssim: %.4f, psnr: %.4f \n) ' % (ssim_output, psnr_output)
message += '---------------------------------------------------------'

print(message)  # print the message
filename = os.path.join(web_dir, 'metrics.txt')

with open(filename, "a") as metric_file:
    metric_file.write('%s\n' % message)  # save the message


