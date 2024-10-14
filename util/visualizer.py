import numpy as np
import os
import ntpath
from util import util
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from collections import OrderedDict
import wandb



class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.name = opt.name
        self.log_dir = opt.log_dir
        self.saved = False
        self.visual_projection_depth = opt.visual_projection_depth
        if not os.path.exists(self.log_dir):
            print('creating the train log directory %s...' % self.log_dir)
            os.makedirs(self.log_dir)

        self.wandb = wandb.init(
            project = "NeuroClear", 
            name = self.name,
            dir = self.log_dir, config = self.opt
        )

        self.wandb.define_metric("epoch")
        self.wandb.define_metric("iters")


    def define_metrics(self, loss_names):
        self.wandb.define_metric("iter")
        for loss_name in loss_names:
            self.wandb.define_metric(loss_name, step_metric = "iter")


    def display_current_results(self, visuals, iter, commit=True):
        """
        Display current results on Weights and Bias.

        Parameters:
            visuals (OrderedDict) -- dictionary of images to display or save.
            iter (int) -- the current iteration 
        """
        image_dict = {}

        for label, image_tensor in visuals.items():

            image = util.tensor2im(image_tensor, imtype=np.uint8)
            b, c, d, h, w = image.shape
            slice_portion = int(d/2) # For 3D images, get three images at increasing depth
            img_xy = image[0, 0, slice_portion, :,:] # (H,W) choose the first sample in the batch
            img_xz = image[0, 0, :, slice_portion, :] # choose the second sample in the batch
            img_yz = image[0, 0, :, :, slice_portion] # choose the third sample in the batch

            img_slice = np.concatenate((img_xy, img_xz, img_yz), axis=1)
            # label_xy = label + '_xy-slice'
            # label_xz = label + '_xz-slice'
            # label_yz = label + '_yz-slice'

            # image_dict[label_xy] = wandb.Image(img_xy, caption = f'label: {label_xy}, step:{iter}')
            # image_dict[label_xz] = wandb.Image(img_xz, caption = f'label: {label_xz}, step:{iter}')
            # image_dict[label_yz] = wandb.Image(img_yz, caption = f'label: {label_yz}, step:{iter}')            

            image_dict[label + '_Slice_img'] = wandb.Image(img_slice, caption = f'{label} slice: xy,xz,yz, iter:{iter}')                       
            # MIP depth for visualization is 20 slices
            gap = int(self.visual_projection_depth/2)
            img_mip_xy = np.amax(image[0, 0, slice_portion-gap:slice_portion+gap, :, :], 0)
            img_mip_xz = np.amax(image[0, 0, :, slice_portion-gap:slice_portion+gap, :], 1)
            img_mip_yz = np.amax(image[0, 0, :, :, slice_portion-gap:slice_portion+gap], 2)
                     
            label_mip_xy = label + '_xy-mip'
            label_mip_xz = label + '_xz-mip'
            label_mip_yz = label + '_yz-mip'

            # image_dict[label_mip_xy] = wandb.Image(img_mip_xy, caption = f'label: {label_mip_xy}, step:{iter}')
            # image_dict[label_mip_xz] = wandb.Image(img_mip_xz, caption = f'label: {label_mip_xz}, step:{iter}')
            # image_dict[label_mip_yz] = wandb.Image(img_mip_yz, caption = f'label: {label_mip_yz}, step:{iter}')  
            
            img_mip = np.concatenate((img_mip_xy, img_mip_xz, img_mip_yz), axis=1)
            image_dict[label + '_Proj_img'] = wandb.Image(img_mip, caption = f'{label} mip: xy,xz,yz, iter:{iter}')                       

        self.wandb.log({'iter': iter} | image_dict, step = iter, commit=commit) # type: ignore


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

    # def plot_current_losses(self, losses, iter, step=None, commit=True):
    #     # Note that you need to add a global point (like step or iteration or epoch count)
    #     self.wandb.log(losses | {'iter':iter}, step=step, commit=commit)


    def plot_current_losses(self, losses, iter, step=None, commit=True):
        # Note that you need to add a global point (like step or iteration or epoch count)
        self.wandb.log( {'iter':iter} | losses, step=iter, commit=commit)

        # for label, loss in losses.items():
        #     if is_epoch:
        #         self.tb_writer.add_scalar('train_by_epoch/' + label, loss, plot_count)
                
        #     else:
        #         self.tb_writer.add_scalar('train_by_iter/' + label, loss, plot_count)