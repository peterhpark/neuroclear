import torch
import itertools
import numpy as np
from .base_model import BaseModel
from . import networks

class AxialToLateralGANDryopsModel(BaseModel):
    """
    NOTE: This is the Apollo model for an ablation case with no backward path.

    G_A: original -> isotropic

    D_A_axial: original_XY <-> isotropic_axial_MIP
    D_A_lateral: original_XY <-> isotropic_XY_MIP

    We only consider one path: A->B.
    We also do not consider buffering fake images for discriminator.

    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--gan_mode', type=str, default='vanilla',
                                help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')

            parser.add_argument('--lambda_plane', type=int, nargs='+', default=[1, 1, 1],
                                help='weight ratio for matching (target vs. target) and (target vs. source) and (MIP target vs. MIP source).')

            parser.add_argument('--randomize_projection_depth', action='store_true', help='randomize the depth for MIP')
            parser.add_argument('--projection_depth', type=int, default=10,
                                help='depth for maximum intensity projections. ')
            parser.add_argument('--min_projection_depth', type=int, default=2,
                                help='minimum depth for maximum intensity projections. ')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A_lateral', 'D_A_axial', 'G_A', 'G_A_lateral', 'G_A_axial']
        self.gan_mode = opt.gan_mode

        self.gen_dimension = 3  # 3D convolutions in generators
        self.dis_dimension = 2  # 2D convolutions in discriminators

        self.randomize_projection_depth = opt.randomize_projection_depth
        if not (self.randomize_projection_depth):
            self.projection_depth_custom = opt.projection_depth
        else:
            self.max_projection_depth = opt.projection_depth
            self.min_projection_depth = opt.min_projection_depth
            print("Projection depth is randomized with maximum depth of %d." % (self.max_projection_depth))

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real', 'fake']

        self.lambda_plane_target, self.lambda_slice, self.lambda_proj = [
            factor / (opt.lambda_plane[0] + opt.lambda_plane[1] + opt.lambda_plane[2]) for factor in opt.lambda_plane]

        self.visual_names = visual_names_A   # combine visualizations for A and B

        self.lateral_axis = 0  # XY plane
        self.axial_1_axis = 1 # XZ plane
        self.axial_2_axis = 2 # YZ plane

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'D_A_lateral', 'D_A_axial']
        else:  # during test time, only load Gs
            self.model_names = ['G_A']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G),  D_A (D_Y)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                        dimension=self.gen_dimension)

        if self.isTrain:  # define discriminators
            self.netD_A_axial = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                  opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, False,
                                                  self.gpu_ids, dimension=self.dis_dimension)

            self.netD_A_lateral = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                    opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, False,
                                                    self.gpu_ids, dimension=self.dis_dimension)


        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A_axial.parameters(), self.netD_A_lateral.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
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
        self.real = input['A' if AtoB else 'B'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        self.cube_shape = self.real.shape
        self.num_slice = self.cube_shape[-3]

        if not (self.randomize_projection_depth):
            self.projection_depth = self.projection_depth_custom
        else:
            self.projection_depth = np.random.randint(max(2, self.min_projection_depth), self.max_projection_depth + 1)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>.
        In this version, we iterate through each slice in a cube.
        """
        self.fake = self.netG_A(self.real)  # G_A(A)

    def backward_D_slice(self, netD, real, fake, slice_axis_real, slice_axis_fake):

        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """

        # Real
        pred_real = self.iter_f(real, netD, slice_axis_real)
        pred_fake = self.iter_f(fake.detach(), netD, slice_axis_fake)

        # real
        loss_D_real = self.criterionGAN(pred_real, True)  # Target_is_real -> True: loss (pred_real - unit vector)

        # Fake
        loss_D_fake = self.criterionGAN(pred_fake, False)  # no loss with the unit vector

        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        loss_D.backward()
        return loss_D

    def backward_D_projection(self, netD, real, fake, slice_axis_real, slice_axis_fake):

        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """

        # Real
        pred_real = self.iter_f(real, netD, slice_axis_real)
        pred_fake = self.proj_f(fake.detach(), netD, slice_axis_fake)

        # real
        loss_D_real = self.criterionGAN(pred_real, True)  # Target_is_real -> True: loss (pred_real - unit vector)

        # Fake
        loss_D_fake = self.criterionGAN(pred_fake, False)  # no loss with the unit vector

        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        loss_D.backward()
        return loss_D

    def backward_D_A_lateral(self):
        self.loss_D_A_lateral = self.backward_D_projection(self.netD_A_lateral, self.real, self.fake, self.lateral_axis,
                                                      self.lateral_axis)  # comparing XY_original to XY_fake_MIP

    def backward_D_A_axial(self): # compares real XY slice image and fake axial MIP image.
        """Calculate GAN loss for discriminator D_A"""
        self.loss_D_A_axial_1 = self.backward_D_projection(self.netD_A_axial, self.real, self.fake, self.lateral_axis,
                                                      self.axial_1_axis)  # comparing XY_original to YZ_fake

        self.loss_D_A_axial_2 = self.backward_D_projection(self.netD_A_axial, self.real, self.fake, self.lateral_axis,
                                                      self.axial_2_axis)

        self.loss_D_A_axial = (self.loss_D_A_axial_1 + self.loss_D_A_axial_2)*0.5

    def backward_G(self):
        """Calculate the loss for generators G_A"""
        self.loss_G_A_lateral = self.criterionGAN(self.proj_f(self.fake, self.netD_A_lateral, self.lateral_axis),
                                                  True) * self.lambda_plane_target

        self.loss_G_A_axial = self.criterionGAN(self.proj_f(self.fake, self.netD_A_axial, self.axial_1_axis),
                                                True) * self.lambda_slice + \
                              self.criterionGAN(self.proj_f(self.fake, self.netD_A_axial, self.axial_2_axis),
                                                True) * self.lambda_slice

        self.loss_G_A = self.loss_G_A_lateral + self.loss_G_A_axial * 0.5

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.

        # G_A
        self.set_requires_grad(
            [self.netD_A_lateral, self.netD_A_axial], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A's gradients to zero
        self.backward_G()  # calculate gradients for G_A
        self.optimizer_G.step()  # update G_A's weights

        # D_A
        self.set_requires_grad(
            [self.netD_A_lateral, self.netD_A_axial], True)
        self.optimizer_D.zero_grad()  # set D_A's gradients to zero

        self.backward_D_A_lateral()
        self.backward_D_A_axial()  # calculate gradients for D_A's

        self.optimizer_D.step()  # update D_A's weights

    # Apply discriminator to each slice in a given dimension and save it as a volume.
    def iter_f(self, input, function, slice_axis):
        input_tensor = Volume(input, self.device)  # Dimension: batch, color_channel, z, y, x
        img_slice = input_tensor.get_slice(slice_axis)  # get image dimension after convolving through the discriminator
        output_slice = function(img_slice)
        return output_slice

    def proj_f(self, input, function, slice_axis):
        input_volume = Volume(input, self.device)
        mip = input_volume.get_projection(self.projection_depth, slice_axis)
        output_mip = function(mip)
        return output_mip

class Volume():
    def __init__(self, vol, device):
        self.volume = vol.to(device)  # push the volume to cuda memory
        self.num_slice = vol.shape[-1]

    # returns a slice: # batch, color_channel, y, x
    def get_slice(self, slice_axis):
        slice_index_pick = np.random.randint(self.num_slice)
        if slice_axis == 0:
            return self.volume[:, :, slice_index_pick, :, :]

        elif slice_axis == 1:
            return self.volume[:, :, :, slice_index_pick, :]

        elif slice_axis == 2:
            return self.volume[:, :, :, :, slice_index_pick]

    def get_projection(self, depth, slice_axis):
        start_index = np.random.randint(0, self.num_slice - depth)
        if slice_axis == 0:
            volume_ROI = self.volume[:, :, start_index:start_index + depth, :, :]

        elif slice_axis == 1:
            volume_ROI = self.volume[:, :, :, start_index:start_index + depth, :]

        elif slice_axis == 2:
            volume_ROI = self.volume[:, :, :, :, start_index:start_index + depth]

        mip = torch.max(volume_ROI, slice_axis + 2)[0]
        return mip

    def get_volume(self):
        return self.volume