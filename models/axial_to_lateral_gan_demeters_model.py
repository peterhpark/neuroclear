# TODO August 06 version
import torch
import itertools
import numpy as np
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class AxialToLateralGANDemetersModel(BaseModel):
    """
    This class implements the CycleGAN model with cubes, for learning image-to-image translation without paired data.

    The model takes a 3D image cube as an input and outputs a 3D image stack that correspond to the output cube.
    Note that the loss functions are readjusted for cube dataset.

    GAN Loss is calculated in 2D between axial image and lateral image. -> Discriminator takes 2D images
                                                                        -> Generator takes 3D images.

    G_A: original -> isotropic
    G_B: isotropic -> original

    D_A_yz: original_XY <-> isotropic_YZ
    D_A_xy: original_XY <-> isotropic_XY
    D_A_xz: original_XY <-> isotropic_XZ

    D_B_yz: original_YZ <-> reconstructed_YZ
    D_B_xy: original_XY <-> reconstructed_XY
    D_B_xz: original XZ <-> reconstructed_XZ

    We only consider one path: A->B.
    We also do not consider buffering fake images for discriminator.

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        We only consider forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Dropout is not used in the original CycleGAN paper.

        D_A compares original volume and isotropic, fake volume.
        D_A_YZ: original_XY <-> isotropic_YZ
        D_A_XY: original_XY <-> isotropic_XY

        D_B compares original volume and reconstructed volume.
        D_B_YZ: original_YZ <-> reconstructed_YZ
        D_B_XY: original_XY <-> reconstructed_XY
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--pool_size', type=int, default=50,
                                help='the size of image buffer that stores previously generated images')
            parser.add_argument('--gan_mode', type=str, default='vanilla',
                                help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')

            parser.add_argument('--randomize_projection_depth', action='store_true', help='randomize the depth for MIP')
            parser.add_argument('--projection_depth', type=int, default=10,
                                help='depth for maximum intensity projections. ')
            parser.add_argument('--min_projection_depth', type=int, default=2,
                                help='minimum depth for maximum intensity projections. ')

        # conversion axis
        parser.add_argument('--lambda_plane', type =int, nargs = '+', default = [1, 1, 1], help = 'weight ratio for matching to source, target, reference plane of fake to target plane of real')

        parser.add_argument('--netG_B', type=str, default='deep_linear_gen', help='Specify the generator in B->A path. ')

        # parser.set_defaults(norm='instance')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A_xy', 'D_A_xz', 'D_A_yz', 'G_A', 'G_A_xy', 'G_A_xz', 'G_A_yz', 'cycle_A', 'D_B_xy', 'D_B_xz', 'D_B_yz',
                           'G_B', 'G_B_xy', 'G_B_xz', 'G_B_yz']
        self.gan_mode = opt.gan_mode

        self.gen_dimension = 3 # 3D convolutions in generators
        self.dis_dimension = 2 # 2D convolutions in discriminators


        self.randomize_projection_depth = opt.randomize_projection_depth
        if not (self.randomize_projection_depth):
            self.projection_depth_custom = opt.projection_depth
        else:
            self.max_projection_depth = opt.projection_depth
            self.min_projection_depth = opt.min_projection_depth
            print("Projection depth is randomized with maximum depth of %d." % (self.max_projection_depth))

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real', 'fake', 'rec']
        visual_names_B = ['real', 'fake', 'rec']

        self.source_sl_axis = 0
        self.target_sl_axis = 1
        self.remain_sl_axis = 2

        self.lambda_plane_target, self.lambda_plane_source, self.lambda_plane_ref = [factor/(opt.lambda_plane[0]+opt.lambda_plane[1]+opt.lambda_plane[2]) for factor in opt.lambda_plane]

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A_xy', 'D_A_xz', 'D_A_yz', 'D_B_xy', 'D_B_xz', 'D_B_yz']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, dimension=self.gen_dimension)

        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG_B, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, dimension=self.gen_dimension)

        if self.isTrain:  # define discriminators
            self.netD_A_yz = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, False, self.gpu_ids, dimension=self.dis_dimension)

            self.netD_A_xy = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, False, self.gpu_ids, dimension=self.dis_dimension)

            self.netD_A_xz = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, False, self.gpu_ids, dimension=self.dis_dimension)

            self.netD_B_yz = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, False, self.gpu_ids, dimension=self.dis_dimension)

            self.netD_B_xy = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, False, self.gpu_ids, dimension=self.dis_dimension)

            self.netD_B_xz = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, False,
                                               self.gpu_ids, dimension=self.dis_dimension)


        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A_yz.parameters(), self.netD_A_xy.parameters(), self.netD_A_xz.parameters(),
                                                                self.netD_B_yz.parameters(), self.netD_B_xy.parameters(), self.netD_B_xz.parameters()),
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
        self.rec = self.netG_B(self.fake)   # G_B(G_A(A))

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
        loss_D_real = self.criterionGAN(pred_real, True) # Target_is_real -> True: loss (pred_real - unit vector)

        # Fake
        loss_D_fake = self.criterionGAN(pred_fake, False) # no loss with the unit vector

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
    
    def backward_D_A_xy(self):
        self.loss_D_A_xy = self.backward_D_projection(self.netD_A_xy, self.real, self.fake, self.target_sl_axis, self.target_sl_axis) # comparing XY_original to XY_fake

    def backward_D_A_yz(self):
        """Calculate GAN loss for discriminator D_A"""
        self.loss_D_A_yz = self.backward_D_projection(self.netD_A_yz, self.real, self.fake, self.target_sl_axis, self.source_sl_axis) # comparing XY_original to YZ_fake

    def backward_D_A_xz(self):
        self.loss_D_A_xz = self.backward_D_projection(self.netD_A_xz, self.real, self.fake, self.target_sl_axis, self.remain_sl_axis) # comparing XY_original to XZ_fake


    def backward_D_B_xy(self):
        self.loss_D_B_xy = self.backward_D_slice(self.netD_B_xy, self.real, self.rec, self.target_sl_axis, self.target_sl_axis) # comparing XY_original to XY_reconstructed

    def backward_D_B_yz(self):
        """Calculate GAN loss for discriminator D_B, which compares the original and the reconstructed. """
        self.loss_D_B_yz = self.backward_D_slice(self.netD_B_yz, self.real, self.rec, self.source_sl_axis, self.source_sl_axis) #comparing YZ_original to YZ_reconstructed

    def backward_D_B_xz(self):
        self.loss_D_B_xz = self.backward_D_slice(self.netD_B_xz, self.real, self.rec, self.remain_sl_axis, self.remain_sl_axis) # comparing XZ_original to XZ_reconstructed

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_A = self.opt.lambda_A

        self.loss_G_A_xy = self.criterionGAN(self.proj_f(self.fake, self.netD_A_xy, self.target_sl_axis), True) * (1/3)
        self.loss_G_A_yz = self.criterionGAN(self.proj_f(self.fake, self.netD_A_yz, self.source_sl_axis), True) * (1/3)
        self.loss_G_A_xz = self.criterionGAN(self.proj_f(self.fake, self.netD_A_xz, self.remain_sl_axis), True) * (1/3)
        self.loss_G_A = self.loss_G_A_xy + self.loss_G_A_yz + self.loss_G_A_xz

        self.loss_G_B_xy = self.criterionGAN(self.iter_f(self.rec, self.netD_B_xy, self.target_sl_axis), True) * (1 / 3)
        self.loss_G_B_yz = self.criterionGAN(self.iter_f(self.rec, self.netD_B_yz, self.source_sl_axis), True) * (1 / 3)
        self.loss_G_B_xz = self.criterionGAN(self.iter_f(self.rec, self.netD_B_xz, self.remain_sl_axis), True) * (1 / 3)

        self.loss_G_B = self.loss_G_B_xy + self.loss_G_B_yz + self.loss_G_B_xz

        # This model only includes forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec, self.real) * lambda_A

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A_xy, self.netD_A_yz, self.netD_A_xz,  self.netD_B_xy, self.netD_B_yz, self.netD_B_xz], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights

        # D_A and D_B
        self.set_requires_grad([self.netD_A_xy, self.netD_A_yz, self.netD_A_xz, self.netD_B_xy, self.netD_B_yz,  self.netD_B_xz], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero

        self.backward_D_A_xy()
        self.backward_D_A_yz()  # calculate gradients for D_A's
        self.backward_D_A_xz()

        self.backward_D_B_xy()
        self.backward_D_B_yz()  # calculate gradients for D_B's
        self.backward_D_B_xz()
        self.optimizer_D.step()  # update D_A and D_B's weights

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