import torch
import itertools
import numpy as np
from .base_model import BaseModel
from . import networks

class SurtModel(BaseModel):
    """
    This model uses high-resolution reference from another source.
    The model takes a 3D image cube as an input and outputs a 3D image stack that correspond to the output cube.
    Note that the loss functions are readjusted for cube dataset.

    This model is a successor to sif; it applies the following additional constraints:
    1. Low frequency preservation by Radon transform. sum(source img, axis= each axis) = sum(generated img, axis= each axis). 
    2. Isotropy preservation. slice(generated img, axis= axial) = slice(generated img, axis= lateral). Instead of matching the axial to the lateral of the target, we match it to the generated image. 

    GAN Loss is calculated in 2D between axial image and lateral image. -> Discriminator takes 2D images
                                                                        -> Generator takes 3D images.

    G_A: original -> high-resolution isotropic
    G_B: high-resolution isotropic -> original

    D_A_axial: isotropic_lateral_MIP <-> isotropic_axial_MIP
    D_A_lateral: ref_XY <-> isotropic_lateral_MIP

    D_B_axial: original_axial <-> reconstructed_axial
    D_B_lateral: original_lateral <-> reconstructed_lateral

    We only consider one path: A->B.
    We also do not consider buffering fake images for discriminator.

    """
    
    def __init__(self, opt):
        """ The Tuisto Class 

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseModel.__init__(self, opt)

        # if opt.data_gt is not None:
        #     self.validate = True
        # else:
        #     self.validate = False

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A_lateral', 'D_A_axial_proj', 'G_A', 'G_A_lateral', 'G_A_axial', 'cycle', 'radon',
                           'D_B_lateral', 'D_B_axial', 'G_B', 'G_B_lateral', 'G_B_axial', 'cycle_B']

        self.gan_mode = opt.gan_mode

        self.gen_dimension = 3  # 3D convolutions in generators
        self.dis_dimension = 2  # 2D convolutions in discriminators

        self.randomize_projection_depth = opt.randomize_projection_depth

        if not (self.randomize_projection_depth):
            self.projection_depth_custom = opt.projection_depth
        else:
            self.max_projection_depth = opt.projection_depth
            self.min_projection_depth = 2

        self.sample_proj = opt.projection_sampling # how many times do we project?
        self.sample_slice = opt.slice_sampling # how many times do we slice?

        self.lambda_radon = opt.lambda_radon
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_tgt', 'real_src', 'fake', 'rec']
        if self.lambda_radon > 0:
            self.visual_names += ['real_src_lateral_radon', 'real_src_axial_radon', 'fake_lateral_radon', 'fake_axial_radon']

        # if self.validate:
        #     self.loss_names += ['valL1', 'valssim']
        #     self.visual_names += ['real_gt']

        self.lambda_plane_target, self.lambda_slice, self.lambda_proj = [
            factor / (opt.lambda_plane[0] + opt.lambda_plane[1] + opt.lambda_plane[2]) for factor in opt.lambda_plane]

        if 'torchioChannelorder' in opt.preprocess:
            # Use TorhIO's dimension ordering: (X, Y, Z)
            self.lateral_axis = 2
            self.axial_1_axis = 0
            self.axial_2_axis = 1
            self.torchio_channelorder = True
        else:
            # Default dimension ordering: (Z, Y, X)
            self.lateral_axis = 0 # XY plane
            self.axial_1_axis = 1 # XZ plane
            self.axial_2_axis = 2 # YZ plane

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A_lateral', 'D_A_axial', 'D_B_lateral', 'D_B_axial']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                        dimension=self.gen_dimension, use_sigmoid=opt.use_sigmoid_inG)

        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG_B, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                        dimension=self.gen_dimension, use_sigmoid=opt.use_sigmoid_inG)

        if self.isTrain:  # define discriminators
            self.netD_A_axial = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                  opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, False,
                                                  self.gpu_ids, dimension=self.dis_dimension)

            self.netD_A_axial_slice = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                    opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, False,
                                                    self.gpu_ids, dimension = self.dis_dimension)
            
            self.netD_A_lateral = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                    opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, False,
                                                    self.gpu_ids, dimension=self.dis_dimension)
            
            self.netD_A_lateral_slice = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                    opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, False,
                                                    self.gpu_ids, dimension = self.dis_dimension)

            self.netD_B_axial = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                                  opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, False,
                                                  self.gpu_ids, dimension=self.dis_dimension)

            self.netD_B_lateral = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                                    opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, False,
                                                    self.gpu_ids, dimension=self.dis_dimension)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionRadon = torch.nn.L1Loss() # Radon transform loss to preserve the low-frequency information.

            # if self.validate:
            #     self.criterionValL1 = torch.nn.L1Loss() # comparison with GT for validation
            #     self.criterionValssim =StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A_axial.parameters(), self.netD_A_lateral.parameters(),
                                self.netD_B_axial.parameters(), self.netD_B_lateral.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        ## END OF INITIALIZATION ##

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_src = input['src' if AtoB else 'tgt'].to(self.device) # 3D image (B, C, Z, Y, X)
        self.real_tgt = input['tgt' if AtoB else 'src'].to(self.device) # 2D image (B, C, Y, X)

        # if self.validate:
        #     self.real_gt = input['gt'].to(self.device)
        #     self.image_paths_gt = input['gt_paths']

        self.image_paths_src = input['src_paths' if AtoB else 'tgt_paths']
        self.image_paths_tgt = input['tgt_paths' if AtoB else 'src_paths']

        self.cube_shape = self.real_src.shape
        self.num_slice = self.cube_shape[-3]

        if not (self.randomize_projection_depth):
            self.projection_depth = self.projection_depth_custom
        else:
            self.projection_depth = np.random.randint(max(1, self.min_projection_depth), self.max_projection_depth + 1)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>.
        In this version, we iterate through each slice in a cube.
        """
        self.fake = self.netG_A(self.real_src)  # G_A(A)
        self.rec = self.netG_B(self.fake)  # G_B(G_A(A))
        self.fake_2 = self.netG_A(self.rec) # fake version of the original fake volume 
    
    
    def backward_D(self, netD, real, fake, slice_axis_real, slice_axis_fake, real_f, fake_f):

        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
            slice_axis_real (int) -- the axis along which the real images are sliced
            slice_axis_fake (int) -- the axis along which the fake images are sliced
            real_f (function) -- the function to apply to each real slice or projection
            fake_f (function) -- the function to apply to each fake slice or projection

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """

        if real.ndim == 4: # 2D image
            pred_real = netD(real.detach())
        else:
            pred_real = real_f(real.detach(), netD, slice_axis_real)

        pred_fake = fake_f(fake.detach(), netD, slice_axis_fake)

        # real
        loss_D_real = self.criterionGAN(pred_real, True)  # Target_is_real -> True: loss (pred_real - unit vector)

        # Fake
        loss_D_fake = self.criterionGAN(pred_fake, False)  # no loss with the unit vector

        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    
    def backward_D_A_lateral(self):
        self.loss_D_A_lateral = self.backward_D(self.netD_A_lateral, self.real_tgt, self.fake, self.lateral_axis, self.lateral_axis, self.slice_f, self.proj_f)  # match fake lateral proj to target lateal slice
    
    def backward_D_A_axial(self): # compares real_tgt XY slice image and fake axial MIP image.
        """Calculate GAN loss for discriminator D_A"""
        self.loss_D_A_axial_1_proj = self.backward_D(self.netD_A_axial, self.fake, self.fake, self.lateral_axis, self.axial_1_axis, self.proj_f, self.proj_f)  # match fake axial to fake lateral
        self.loss_D_A_axial_2_proj = self.backward_D(self.netD_A_axial, self.fake, self.fake, self.lateral_axis, self.axial_1_axis, self.proj_f, self.proj_f)  # match fake axial to fake lateral
        self.loss_D_A_axial_proj = (self.loss_D_A_axial_1_proj + self.loss_D_A_axial_2_proj)*0.5

        # # match fake lateral slice to fake axial slice
        # self.loss_D_A_texturematch_axial_1 = self.backward_D(self.netD_A_lateral_slice, self.fake, self.fake, self.lateral_axis, self.axial_1_axis, self.slice_f, self.slice_f)  # match fake lateral slice to target lateral slice

    def backward_D_B_lateral(self):
        self.loss_D_B_lateral = self.backward_D(self.netD_B_lateral, self.real_src, self.rec, self.lateral_axis,
                                                      self.lateral_axis, self.proj_f, self.proj_f)  # comparing XY_original to XY_reconstructed

    def backward_D_B_axial(self): # compares real_tgt axial slice image and fake axial slice image.
        """Calculate GAN loss for discriminator D_B, which compares the original and the reconstructed. """
        self.loss_D_B_axial_1 = self.backward_D(self.netD_B_axial, self.real_src, self.rec, self.axial_1_axis,
                                                      self.axial_1_axis, self.proj_f, self.proj_f)  # comparing YZ_original to YZ_reconstructed

        self.loss_D_B_axial_2 = self.backward_D(self.netD_B_axial, self.real_src, self.rec, self.axial_2_axis,
                                                      self.axial_2_axis, self.proj_f, self.proj_f)  # comparing YZ_original to YZ_reconstructed

        self.loss_D_B_axial = (self.loss_D_B_axial_1 + self.loss_D_B_axial_2)*0.5

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_A = self.opt.lambda_A

        self.loss_G_A_lateral = self.criterionGAN(self.proj_f(self.fake, self.netD_A_lateral, self.lateral_axis),
                                                  True) * self.lambda_plane_target

        self.loss_G_A_axial = self.criterionGAN(self.proj_f(self.fake, self.netD_A_axial, self.axial_1_axis),
                                                True) * self.lambda_slice + \
                              self.criterionGAN(self.proj_f(self.fake, self.netD_A_axial, self.axial_2_axis),
                                                True) * self.lambda_slice
        

        self.loss_G_A = self.loss_G_A_lateral + self.loss_G_A_axial * 0.5

        self.loss_G_B_lateral = self.criterionGAN(self.proj_f(self.rec, self.netD_B_lateral, self.lateral_axis),
                                                  True) * self.lambda_plane_target
        self.loss_G_B_axial = self.criterionGAN(self.proj_f(self.rec, self.netD_B_axial, self.axial_1_axis),
                                                True) * self.lambda_slice + \
                              self.criterionGAN(self.proj_f(self.rec, self.netD_B_axial, self.axial_2_axis),
                                                True) * self.lambda_slice

        self.loss_G_B = self.loss_G_B_lateral + self.loss_G_B_axial * 0.5

        # Cycle Consistency Loss
        self.loss_cycle = self.criterionCycle(self.rec, self.real_src) * lambda_A 
        self.loss_cycle_B = self.criterionCycle(self.fake, self.fake_2) * lambda_A

        if self.lambda_radon > 0:
            # Radon Transform Loss
            self.real_src_lateral_radon = self.sum_f(self.real_src, self.lateral_axis)
            self.fake_lateral_radon = self.sum_f(self.fake, self.lateral_axis)
            self.real_src_axial_radon = self.sum_f(self.real_src, self.axial_1_axis)
            self.fake_axial_radon = self.sum_f(self.fake, self.axial_1_axis)

            self.loss_radon = self.criterionRadon(self.fake_lateral_radon, self.real_src_lateral_radon) * self.lambda_radon + \
                                        self.criterionRadon(self.fake_axial_radon, self.real_src_axial_radon) * self.lambda_radon + \
                                        self.criterionRadon(self.sum_f(self.fake, self.axial_2_axis), self.sum_f(self.real_src, self.axial_2_axis)) * self.lambda_radon
            
        else:
            self.loss_radon = 0 
        
        # if self.validate:
        #     # calculate validation losses
        #     self.loss_valL1 = self.criterionValL1(self.fake.detach(), self.real_gt)
        #     self.loss_valssim = self.criterionValssim(self.fake.detach(), self.real_gt)

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle + self.loss_cycle_B + self.loss_radon
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.

        # G_A and G_B
        self.set_requires_grad(
            [self.netD_A_lateral, self.netD_A_axial, self.netD_B_lateral, self.netD_B_axial], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights

        # D_A and D_B
        self.set_requires_grad(
            [self.netD_A_lateral, self.netD_A_axial, self.netD_B_lateral, self.netD_B_axial], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero

        self.backward_D_A_lateral()
        self.backward_D_A_axial()  # calculate gradients for D_A's

        self.backward_D_B_lateral()
        self.backward_D_B_axial()  # calculate gradients for D_B's
        self.optimizer_D.step()  # update D_A and D_B's weights

    def proj_f(self, input, function, slice_axis):

        """
        Parameters:
            self (SurtModel): The instance of the SurtModel class.
            input (tensor): The input volume tensor.
            function (callable): The function to apply to each slice.
            slice_axis (int): The axis along which to slice the volume.

        Returns:
            tensor: The output tensor after applying the function to each projection.
        """
          
        input_volume = Volume(input, self.device)
        output_list = []

        num_slice_per_axis = input_volume.volume.shape[slice_axis + 2]  
        sampled_indices = np.random.choice(num_slice_per_axis-self.projection_depth, self.sample_proj, replace=False)

        for index in sampled_indices: 
            mip = input_volume.get_projection(index, self.projection_depth, slice_axis)
            output_mip = function(mip)
            output_list.append(output_mip)

        output = torch.stack(output_list, dim=2) # Dimension: batch, color_channel, sample_proj, dis_y, dis_x
        return output 
    
    def slice_f(self, input, function, slice_axis):
        # Dimension: batch, color_channel, z, y, x
        """
        Parameters:
            self (SurtModel): The instance of the SurtModel class.
            input (tensor): The input volume tensor.
            function (callable): The function to apply to each slice.
            slice_axis (int): The axis along which to slice the volume.

        Returns:
            tensor: The output tensor after applying the function to each slice.
        """

        input_volume = Volume(input, self.device) # Dimension: batch, color_channel, z, y, x
        output_list = []
        num_slice_per_axis = input_volume.volume.shape[slice_axis + 2]
        sampled_indices = np.random.choice(num_slice_per_axis, self.sample_slice, replace=False)

        for index in sampled_indices:
            slice_img = input_volume.get_slice(index, slice_axis)
            output_slice = function(slice_img)
            output_list.append(output_slice)

        output = torch.stack(output_list, dim=2) # Dimension: batch, color_channel, sample_proj, dis_y, dis_x
        return output

    def sum_f(self, input, slice_axis):
        """
        Parameters:
            self (SurtModel): The instance of the SurtModel class.
            input (tensor): The input volume tensor.
            slice_axis (int): The axis along which to slice the volume.

        Returns:
            tensor: The normalized sum of the input tensor along the specified axis.
        """
        radon_integral = torch.sum(input, slice_axis + 2)
        radon_integral = (radon_integral - radon_integral.min()) / (radon_integral.max() - radon_integral.min())
        return radon_integral

class Volume():
    def __init__(self, vol, device):
        self.volume = vol.to(device)  # push the volume to cuda memory
        self.num_slice_x = vol.shape[-1]
        self.num_slice_y = vol.shape[-2]
        self.num_slice_z = vol.shape[-3]

    def get_projection(self, index, depth, slice_axis):
       
        if slice_axis == 0:
            volume_ROI = self.volume[:, :, index:index + depth, :, :]

        elif slice_axis == 1:
            volume_ROI = self.volume[:, :, :, index:index + depth, :]

        elif slice_axis == 2:
            volume_ROI = self.volume[:, :, :, :, index:index + depth]
        mip = torch.max(volume_ROI, slice_axis + 2)[0] # plus two because first two indices are not relevant.
        return mip

    def get_slice(self, index, slice_axis):
        if slice_axis == 0:
            return self.volume[:, :, index, :, :]
        elif slice_axis == 1:
            return self.volume[:, :, :, index, :]
        elif slice_axis == 2:
            return self.volume[:, :, :, :, index]

    def set_slice(self, index, slice_axis, slice_data):
        if slice_axis == 0:
            self.volume[:, :, index, :, :] = slice_data
        elif slice_axis == 1:
            self.volume[:, :, :, index, :] = slice_data
        elif slice_axis == 2:
            self.volume[:, :, :, :, index] = slice_data

    def get_volume(self):
        return self.volume