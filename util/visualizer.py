# TODO Sep 08 version 
import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
from PIL import Image
import torchvision
import datetime
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.gridspec as gridspec
from collections import OrderedDict
from matplotlib import cm
from tifffile import imsave
import torch

def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk,

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these images (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    # name = image_path

    webpage.add_header(name)
    ims, txts, links = [], [], []
    for label, im_data in visuals.items():
        image_numpy = util.tensor2im(im_data)
        # Need to subtract background to reduce the across-patch background intensity difference.

        label_image_dir = image_dir+'/'+label+'/'
        if not os.path.exists(label_image_dir):
            os.makedirs(label_image_dir)

        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(label_image_dir, image_name)
        image_numpy = image_numpy.squeeze()

        util.save_image(image_numpy, save_path, aspect_ratio=aspect_ratio)
        ims.append(save_path)
        txts.append(label)
        links.append(save_path)
    # TODO: Fix it so that the webpage correctly displays both images.
    webpage.add_images(ims, txts, links, width=width)

def save_test_metrics(save_dir, opt, ssims, psnrs):
    ssim_avg_input_gt = ssims[0]
    ssim_avg_output_gt = ssims[1]
    ssim_whole_input_gt = ssims[2]
    ssim_whole_output_gt = ssims[3]

    psnr_avg_input_gt = psnrs[0]
    psnr_avg_output_gt = psnrs[1]
    psnr_avg_whole_input_gt = psnrs[2]
    psnr_avg_whole_output_gt = psnrs[3]

    message = 'Experiment Name: ' + opt.name + '\n'
    message += '-------------------------------------------------\n'
    message += 'Network Input vs. Groundtruth\n'
    message += '(ssim_avg: %.4f, psnr_avg: %.4f, ssim_whole: %.4f, psnr_whole: %.4f)\n' % (ssim_avg_input_gt, psnr_avg_input_gt, ssim_whole_input_gt, psnr_avg_whole_input_gt)
    message += '-------------------------------------------------\n'
    message += 'Network Output vs. Groundtruth\n'
    message += '(ssim_avg: %.4f, psnr_avg: %.4f, ssim_whole: %.4f, psnr_whole: %.4f)\n' % (ssim_avg_output_gt, psnr_avg_output_gt, ssim_whole_output_gt, psnr_avg_whole_output_gt)
    message += '-------------------------------------------------'

    print(message)  # print the message
    filename = os.path.join(save_dir, 'metrics.txt')

    with open(filename, "a") as metric_file:
        metric_file.write('%s\n' % message)  # save the message


import numpy as np

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.win_size = opt.display_winsize
        self.use_html = opt.isTrain and not opt.no_html
        self.name = opt.name
        self.port = opt.display_port
        self.display_histogram = opt.display_histogram

        self.saved = False

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

        self.tb_dir = os.path.join(opt.checkpoints_dir, 'tensorboard')
        print('create tensorboard directory %s...' % self.tb_dir)
        util.mkdir(self.tb_dir)

        from torch.utils.tensorboard import SummaryWriter
        #TODO: log differently by time.
        self.log_dir  = os.path.join(self.tb_dir, self.name)

        self.tb_writer = SummaryWriter(self.log_dir)

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status."""
        self.saved = False

    # def display_current_model(self, model):
    #     self.tb_writer.add(graph)

    def display_current_results(self, visuals, epoch):
        """
        Display current results on tensorboard.

        Parameters:
            visuals (OrderedDict) -- dictionary of images to display or save.
            epoch (int) -- the current epoch
        """
        for label, image in visuals.items():
            if self.opt.model != 'classifier':
                img_np = util.tensor2im(image, imtype=np.uint8)
                img_shape = img_np.shape
                b, c, d, h, w = img_shape
                slice_portion = int(d/2) # For 3D images, get three images at increasing depth
                img_sample = img_np[0, 0, slice_portion, :,:] # choose the first sample in the batch
                img_sample2 = img_np[0, 0, :, slice_portion, :] # choose the second sample in the batch
                img_sample3 = img_np[0, 0, :, :, slice_portion] # choose the third sample in the batch

                fig_slice = plt.figure(edgecolor='b', dpi=150)
                ax = fig_slice.add_subplot(1, 3, 1)
                ax2 = fig_slice.add_subplot(1, 3, 2)
                ax3 = fig_slice.add_subplot(1, 3, 3)

                ax.set_axis_off()
                ax2.set_axis_off()
                ax3.set_axis_off()

                ax.set_title('XY slice')
                ax2.set_title('XZ slice')
                ax3.set_title('YZ slice')

                ax.imshow(img_sample, cmap='gray')
                ax2.imshow(img_sample2, cmap='gray')
                ax3.imshow(img_sample3, cmap='gray')

                plt.gca().set_axis_off()
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                    hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.close(fig_slice)

                img_mip_xy = np.amax(img_np[0,0], 0)
                img_mip_xz = np.amax(img_np[0,0], 1)
                img_mip_yz = np.amax(img_np[0,0], 2)

                fig_mip = plt.figure(edgecolor='b', dpi=150)
                ax_2_1 = fig_mip.add_subplot(1, 3, 1)
                ax_2_2= fig_mip.add_subplot(1, 3, 2)
                ax_2_3 = fig_mip.add_subplot(1, 3, 3)

                ax_2_1.set_axis_off()
                ax_2_2.set_axis_off()
                ax_2_3.set_axis_off()

                ax_2_1.set_title('XY MIP')
                ax_2_2.set_title('XZ MIP')
                ax_2_3.set_title('YZ MIP')

                ax_2_1.imshow(img_mip_xy, vmax=256, cmap='gray')
                ax_2_2.imshow(img_mip_xz, vmax=256,cmap='gray')
                ax_2_3.imshow(img_mip_yz, vmax=256, cmap='gray')

                plt.gca().set_axis_off()
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                    hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.close(fig_mip)

                self.tb_writer.add_figure('train_slice_images/' + label, fig_slice, epoch)
                self.tb_writer.add_figure('train_mip_images/' + label, fig_mip, epoch)

            else: # if the model is a classifier, display with the labels.
                if label == 'output_tr_softmax' or label == 'output_val_softmax' or label =='label_GT':

                    #image[0] chooses the first item in the batch.
                    predicted = torch.argmax(image[0])
                    label_print = predicted.cpu().float().numpy()
                    if label_print == 0:
                        label_print_str = 'Axial'
                    elif label_print == 1:
                        label_print_str = 'Lateral'

                    fig_slice = plt.figure()
                    plt.text(0.1, 0.4, label_print_str, size=60, bbox=dict(boxstyle="square",
                                                                         ec=(1., 0.5, 0.5),
                                                                         fc=(1., 0.8, 0.8),
                                                                         ))
                    plt.show()
                    plt.close(fig_slice)

                    self.tb_writer.add_figure('train_images/' + label, fig_slice, epoch)

                else:
                    img_np = util.tensor2im(image[0], imtype=np.uint8)
                    img_np = img_np.squeeze()
                    fig_slice = plt.figure()
                    plt.imshow(img_np, cmap='gray')
                    plt.close(fig_slice)

                self.tb_writer.add_figure('train_images/' + label, fig_slice, epoch)

    def display_model_hyperparameters(self): # note that in tensorboard, it is shown as markdowns.
        message = '--------------- Options ------------------  \n'
        for k, v in sorted(vars(self.opt).items()):
            comment = ''
            message += '**{:>1}**: {:>10}{}  \n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        self.tb_writer.add_text('Model_hyperparameters', message)

    def display_current_histogram(self, visuals, epoch):
        for label, image in visuals.items():
            if self.display_histogram:
                self.tb_writer.add_histogram('train_histograms/' + label, image[0][0], epoch)

    def display_graph(self, model, visuals):
        for label, image in visuals.items():
            self.tb_writer.add_graph(model, image)

    def save_current_visuals(self, visuals, epoch):
        for label, image in visuals.items():
            img_np = util.tensor2im(image[0], imtype=np.uint8)
            file_name = os.path.join(self.img_dir, str(epoch) + '_' + str(label)+'.tif')
            imsave(file_name, img_np)

    def plot_current_losses(self, plot_count, losses, is_epoch=False):
        """display the current losses on tensorboard display: dictionary of error labels and values

         Parameters:
            plot_count (int) -- iteration count (default) or epoch count

            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
         """
        for label, loss in losses.items():
            if is_epoch:
                self.tb_writer.add_scalar('train_by_epoch/' + label, loss, plot_count)
            else:
                self.tb_writer.add_scalar('train_by_epoch_progress/' + label, loss, plot_count)

    def print_current_losses(self, epoch, epoch_progress, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            epoch_progress (int) -- current training progress in this epoch in percent (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, epoch_progress: %d%%, iter time: %.3f, data load time: %.3f) ' % (epoch, epoch_progress, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message