import numpy as np
import os
import ntpath
from . import util
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from collections import OrderedDict
from tifffile import imsave
import ntpath
import wandb

def save_images(visuals, save_dir, name = ""):
    """Save images to the disk,

    Parameters:
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """

    img_name = ntpath.basename(name[0])

    for label, im_data in visuals.items():
        image_numpy = util.tensor2im(im_data)

        label_image_dir = save_dir+'/'+label+'/'
        if not os.path.exists(label_image_dir):
            os.makedirs(label_image_dir)

        file_name = '%s_%s.tif' % (img_name, label)
        save_path = os.path.join(label_image_dir, file_name)
        image_numpy = image_numpy.squeeze()
        imsave(save_path, image_numpy)


class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.win_size = opt.display_winsize
        self.use_html = opt.isTrain and not opt.no_html
        self.name = opt.name
        self.log_dir = opt.log_dir
        self.saved = False

        if not os.path.exists(self.log_dir):
            print('creating the train log directory %s...' % self.log_dir)
            os.makedirs(self.log_dir)

        self.wandb = wandb.init(
            project = "NeuroClear", 
            name = self.name,
            dir = self.log_dir
        )

        self.wandb.define_metric("epoch")
        self.wandb.define_metric("train_loss_by_epoch", step_metric = "epoch")
        self.wandb.define_metric("prediction", step_metric = "epoch")
        self.wandb.define_metric("GT", step_metric = "epoch")


    def display_current_results(self, visuals, epoch):
        """
        Display current results on Weights and Bias.

        Parameters:
            visuals (OrderedDict) -- dictionary of images to display or save.
            epoch (int) -- the current epoch
        """
        image_dict = {}

        for label, image_tensor in visuals.items():

            image = util.tensor2im(image_tensor, imtype=np.uint8)
            b, c, d, h, w = image.shape
            slice_portion = int(d/2) # For 3D images, get three images at increasing depth
            img_xy = image[0, 0, slice_portion, :,:] # choose the first sample in the batch
            img_xz = image[0, 0, :, slice_portion, :] # choose the second sample in the batch
            img_yz = image[0, 0, :, :, slice_portion] # choose the third sample in the batch

            label_xy = label + '_xy-slice'
            label_xz = label + '_xz-slice'
            label_yz = label + '_yz-slice'

            image_dict[label_xy] = wandb.Image(img_xy, caption = f'label: {label_xy}, epoch:{epoch}')
            image_dict[label_xz] = wandb.Image(img_xz, caption = f'label: {label_xz}, epoch:{epoch}')
            image_dict[label_yz] = wandb.Image(img_yz, caption = f'label: {label_yz}, epoch:{epoch}')            
                        
            # MIP depth for visualization is 30 slices.
            img_mip_xy = np.amax(image[0, 0, slice_portion-15:slice_portion+15, :, :], 0)
            img_mip_xz = np.amax(image[0, 0, :, slice_portion-15:slice_portion+15, :], 1)
            img_mip_yz = np.amax(image[0, 0, :, :, slice_portion-15:slice_portion+15], 2)
                     
            label_mip_xy = label + '_xy-mip'
            label_mip_xz = label + '_xz-mip'
            label_mip_yz = label + '_yz-mip'

            image_dict[label_mip_xy] = wandb.Image(img_mip_xy, caption = f'label: {label_mip_xy}, epoch:{epoch}')
            image_dict[label_mip_xz] = wandb.Image(img_mip_xz, caption = f'label: {label_mip_xz}, epoch:{epoch}')
            image_dict[label_mip_yz] = wandb.Image(img_mip_yz, caption = f'label: {label_mip_yz}, epoch:{epoch}')  

        self.wandb.log(image_dict | {'epoch': epoch}, commit=True) # type: ignore


    # TODO add if needed 
    # def display_current_histogram(self, visuals, epoch):
    #     for label, image in visuals.items():
    #         image = image.squeeze()
    #         if self.display_histogram:
    #             self.tb_writer.add_histogram('train_histograms/' + label, image, epoch)

    # def display_graph(self, model, visuals):
    #     for label, image in visuals.items():
    #         self.tb_writer.add_graph(model, image)

    # def save_current_visuals(self, visuals, epoch):
    #     image_dict = {}
    #     for label, image in visuals.items():
    #         img_np = util.tensor2im(image[0], imtype=np.uint8)
    #         file_name = os.path.join(self.img_dir, str(epoch) + '_' + str(label)+'.tif')
    #         imsave(file_name, img_np)

    def plot_current_losses(self, losses, step=None, commit=True):
        # Note that you need to add a global point (like step or iteration or epoch count)
        self.wandb.log(losses, step=step, commit=commit)
        
        # for label, loss in losses.items():
        #     if is_epoch:
        #         self.tb_writer.add_scalar('train_by_epoch/' + label, loss, plot_count)
                
        #     else:
        #         self.tb_writer.add_scalar('train_by_iter/' + label, loss, plot_count)