# SEP 8 version
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from util.util import noisy

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance', dimension =3):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """

    if norm_type == 'batch':
        norm_layer = functools.partial(batch_norm(dimension), affine=True, track_running_stats=True)

    elif norm_type == 'instance':
        norm_layer = functools.partial(instance_norm(dimension), affine=False, track_running_stats=False)

    elif norm_type == 'spectral':
        norm_layer = lambda x: Identity()

    elif norm_type == 'none':
        norm_layer = lambda x: Identity()

    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer




def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """

    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif opt.lr_policy == 'constant':
        def lambda_rule(epoch):
            lr_l = 1.0
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)

    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)

    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)

    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)

    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[],
             kernel_size=9, given_psf=None, noise_setting = None, dimension = 3):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm, dimension=dimension)

    if netG == 'unet_twoouts':
        net = UnetTwoOuts(4, output_nc)
    elif netG== 'unet_deconv':
        net = Unet_deconv(1, output_nc, norm_layer=norm_layer, dimension=dimension)
    elif netG== 'unet_vanilla':
        net = Unet_vanilla(1, output_nc, norm_layer=norm_layer, dimension=dimension)
    elif netG == 'unet_classic':
        net = UnetGenerator(input_nc, output_nc, dimension=dimension, norm_layer=norm_layer)
    elif netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'VGG':
        net = VGG_net(input_nc, num_classes=2, VGG_type='VGG16')
    elif netG =='linearkernel':
        net = LinearKernel(input_nc, output_nc, kernel_size, dimension = dimension)
    elif netG =='linearkernel_double':
        net = LinearKernel_double(input_nc, output_nc, kernel_size, dimension=dimension)
    elif netG == 'linearkernel_LK31': # for testing
        net = LinearKernel(input_nc, output_nc, 31, dimension = dimension)
    elif netG == 'linearkernel_NC':
        net = LinearKernel_NC(input_nc, output_nc, kernel_size, dimension=dimension)
    elif netG =='fixed_kernel':
        net = FixedLinearKernel(given_psf, noise_setting)
    elif netG =='deep_linear_gen':
        net = DeepLinearGenerator(input_nc, output_nc)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, use_sigmoid=False,
             gpu_ids=[], dimension =3):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm, dimension=dimension)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, use_sigmoid=use_sigmoid, norm_layer=norm_layer, dimension=dimension)
    elif netD =='basic_SN':
        net = NLayerDiscriminatorSN(input_nc, ndf, n_layers=3, use_sigmoid=use_sigmoid, norm_layer=norm_layer, dimension=dimension)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, use_sigmoid=use_sigmoid, norm_layer=norm_layer, dimension=dimension)
    elif netD == 'n_layers_SN':  # more options
        net = NLayerDiscriminatorSN(input_nc, ndf, n_layers_D, use_sigmoid=use_sigmoid, norm_layer=norm_layer, dimension=dimension)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, dimension=dimension)
    elif netD =='kernelGAN':
        net = KernelPatchDiscriminator(input_nc, ndf, n_layers=5, norm_layer=norm_layer, dimension=dimension)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)

##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label)) # what's the need for a buffer here?
        self.register_buffer('fake_label', torch.tensor(target_fake_label)) # if you make a parameter like this, the value gets stored in the GPU
                                                                            # https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723/7
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif 'wgan' in self.gan_mode:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - typically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label # 1 by default
        else:
            target_tensor = self.fake_label # 0 by default
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and ground truth labels.

        Parameters:
            prediction (tensor) - - typically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif 'wgan' in self.gan_mode:
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """

    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradients_norm = (gradients + 1e-16).norm(2, dim=1)
        # gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        gradient_penalty = ((gradients_norm - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

def conv(dimension):
    if dimension == 2:
        return nn.Conv2d

    elif dimension == 3:
        return nn.Conv3d

    else:
        raise Exception('Invalid image dimension.')

def maxpool(dimension):

    if dimension == 2:
        return nn.MaxPool2d

    elif dimension == 3:
        return nn.MaxPool3d

    else:
        raise Exception('Invalid image dimension.')

def convtranspose(dimension):
    if dimension == 2:
        return nn.ConvTranspose2d

    elif dimension == 3:
        return nn.ConvTranspose3d

    else:
        raise Exception('Invalid image dimension.')


def batch_norm(dimension):
    if dimension == 2:
        return nn.BatchNorm2d

    elif dimension == 3:
        return nn.BatchNorm3d

    else:
        raise Exception('Invalid image dimension.')

def instance_norm(dimension, affine = False):
    if dimension == 2:
        return nn.InstanceNorm2d

    elif dimension == 3:
        return nn.InstanceNorm3d

    else:
        raise Exception('Invalid image dimension.')
###########################PETER'S UNET IMPLEMENTATION##############################################
class double_conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, norm_layer=None, dimension = 3):
        super(double_conv, self).__init__()
        _conv = conv(dimension)

        self.convolution = nn.Sequential(
            _conv(in_channels, out_channels, kernel_size,
                stride, padding),
            norm_layer(out_channels),
            nn.ReLU(),
            _conv(out_channels, out_channels, kernel_size,
                stride, padding),
            norm_layer(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.convolution(x)
        return x

class last_conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, norm_layer=None, dimension = 3):
        super(last_conv, self).__init__()

        _conv = conv(dimension)

        self.convolution = nn.Sequential(
            _conv(in_channels, out_channels, kernel_size,
                stride, padding),
            norm_layer(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.convolution(x)
        return x

class triple_conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, norm_layer=None, dimension = 3):
        super(triple_conv, self).__init__()

        _conv = conv(dimension)

        self.convolution = nn.Sequential(
            _conv(in_channels, out_channels, kernel_size,
                stride, padding),
            norm_layer(out_channels),
            nn.ReLU(),
            _conv(out_channels, out_channels, kernel_size,
                stride, padding),
            norm_layer(out_channels),
            nn.ReLU(),
            _conv(out_channels, out_channels, kernel_size,
                stride, padding),
            norm_layer(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.convolution(x)
        return x

class Unet_deconv(nn.Module):

    def __init__(self, input_nc, output_nc, norm_layer=None, dimension = 3):

        _maxpool = maxpool(dimension)
        _conv = conv(dimension)
        _convtranspose = convtranspose(dimension)

        super(Unet_deconv, self).__init__()
        start_nc = input_nc * 64

        # Downsampling
        self.double_conv1 = double_conv(input_nc,  start_nc, 3, 1, 1, norm_layer, dimension)
        self.maxpool1 = _maxpool(2)

        self.double_conv2 = double_conv(start_nc, start_nc*2, 3, 1, 1, norm_layer, dimension)
        self.maxpool2 = _maxpool(2)

        # bottom floor
        self.bottom_layer = triple_conv(start_nc *2, start_nc*4, 3, 1, 1, norm_layer, dimension)

        # Upsampling = transposed convolution
        self.t_conv2 = _convtranspose (start_nc*4, start_nc*2, 2, 2)
        self.ex_double_conv2 = double_conv(start_nc*4, start_nc*2, 3, 1, 1, norm_layer, dimension)

        self.t_conv1 = _convtranspose(start_nc*2, start_nc, 2, 2)
        self.ex_conv1_1 = last_conv(start_nc*2, start_nc, 3, 1, 1, norm_layer, dimension)

        # last stage
        self.one_by_one = _conv(start_nc, output_nc, 1, 1, 0)
        self.one_by_one_2 = _conv(output_nc, output_nc, 1, 1, 0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):

        # Contracting Path
        conv1 = self.double_conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.double_conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        # bottom floor
        conv_bottom = self.bottom_layer(maxpool2)

        t_conv2 = self.t_conv2(conv_bottom)

        cat2 = torch.cat([conv2, t_conv2], 1)

        ex_conv2 = self.ex_double_conv2(cat2)

        t_conv1 = self.t_conv1(ex_conv2)
        cat1 = torch.cat([conv1, t_conv1], 1)
        ex_conv1 = self.ex_conv1_1(cat1)

        one_by_one = self.one_by_one(ex_conv1)
        one_by_one_2 = self.one_by_one_2(one_by_one)
        last_val = self.sigmoid(one_by_one_2)

        return last_val

class Unet_vanilla(nn.Module):

    def __init__(self, input_nc, output_nc, norm_layer=None, dimension=3):
        _maxpool = maxpool(dimension)
        _conv = conv(dimension)
        _convtrasnpose = convtranspose(dimension)

        super(Unet_vanilla, self).__init__()
        start_nc = input_nc * 64

        # Downsampling
        self.double_conv1 = double_conv(input_nc, start_nc, 3, 1, 1, norm_layer, dimension)
        self.maxpool1 = _maxpool(2)

        self.double_conv2 = double_conv(start_nc, start_nc * 2, 3, 1, 1, norm_layer, dimension)
        self.maxpool2 = _maxpool(2)

        self.double_conv3 = double_conv(start_nc * 2, start_nc * 4, 3, 1, 1, norm_layer, dimension)
        self.maxpool3 = _maxpool(2)

        # bottom floor
        self.bottom_layer = double_conv(start_nc * 4, start_nc * 8, 3, 1, 1, norm_layer, dimension)

        # Upsampling = transposed convolution
        self.t_conv3 = _convtrasnpose(start_nc * 8, start_nc * 4, 2, 2)
        self.ex_double_conv3 = double_conv(start_nc *8, start_nc *4, 3, 1, 1, norm_layer, dimension)

        self.t_conv2 = _convtrasnpose(start_nc * 4, start_nc * 2, 2, 2)
        self.ex_double_conv2 = double_conv(start_nc * 4, start_nc * 2, 3, 1, 1, norm_layer, dimension)

        self.t_conv1 = _convtrasnpose(start_nc * 2, start_nc, 2, 2)
        self.ex_conv1_1 = double_conv(start_nc * 2, start_nc, 3, 1, 1, norm_layer, dimension)

        # last stage
        self.one_by_one = _conv(start_nc, output_nc, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # Contracting Path
        conv1 = self.double_conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.double_conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.double_conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        # bottom floor
        conv_bottom = self.bottom_layer(maxpool3)

        t_conv3 = self.t_conv3(conv_bottom)
        # print ('conv3_shape: ' + str(conv3.shape))
        # print ('t_conv3_shape: ' + str(t_conv3.shape))
        cat3 = torch.cat([conv3, t_conv3], 1)
        ex_conv3 = self.ex_double_conv3(cat3)

        t_conv2 = self.t_conv2(ex_conv3)
        cat2 = torch.cat([conv2, t_conv2], 1)
        ex_conv2 = self.ex_double_conv2(cat2)

        t_conv1 = self.t_conv1(ex_conv2)
        cat1 = torch.cat([conv1, t_conv1], 1)
        ex_conv1 = self.ex_conv1_1(cat1)

        one_by_one = self.one_by_one(ex_conv1)
        last_val = self.sigmoid(one_by_one)

        return last_val

class Unet_vanilla_shallow(nn.Module):

    def __init__(self, input_nc, output_nc, norm_layer=None, dimension=3):
        _maxpool = maxpool(dimension)
        _conv = conv(dimension)
        _convtrasnpose = convtranspose(dimension)

        super(Unet_vanilla, self).__init__()
        start_nc = input_nc * 64

        # Downsampling
        self.double_conv1 = double_conv(input_nc, start_nc, 3, 1, 1, norm_layer, dimension)
        self.maxpool1 = _maxpool(2)

        self.double_conv2 = double_conv(start_nc, start_nc * 2, 3, 1, 1, norm_layer, dimension)
        self.maxpool2 = _maxpool(2)

        # bottom floor
        self.bottom_layer = double_conv(start_nc * 2, start_nc * 4, 3, 1, 1, norm_layer, dimension)

        # Upsampling = transposed convolution
        self.t_conv2 = _convtrasnpose(start_nc * 4, start_nc * 2, 2, 2)
        self.ex_double_conv2 = double_conv(start_nc * 4, start_nc * 2, 3, 1, 1, norm_layer, dimension)

        self.t_conv1 = _convtrasnpose(start_nc * 2, start_nc, 2, 2)
        self.ex_conv1_1 = double_conv(start_nc * 2, start_nc, 3, 1, 1, norm_layer, dimension)

        # last stage
        self.one_by_one = _conv(start_nc, output_nc, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # Contracting Path
        conv1 = self.double_conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.double_conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        # bottom floor
        conv_bottom = self.bottom_layer(maxpool2)
        ex_conv3 = self.ex_double_conv3(conv_bottom)

        t_conv2 = self.t_conv2(ex_conv3)
        cat2 = torch.cat([conv2, t_conv2], 1)
        ex_conv2 = self.ex_double_conv2(cat2)

        t_conv1 = self.t_conv1(ex_conv2)
        cat1 = torch.cat([conv1, t_conv1], 1)
        ex_conv1 = self.ex_conv1_1(cat1)

        one_by_one = self.one_by_one(ex_conv1)
        last_val = self.sigmoid(one_by_one)

        return last_val



VGG_types = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_net(nn.Module):
    """
    VGG network to calculate a perceptual loss.

    Ref#1: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/79f2e1928906f3cccbae6c024f3f79fd05262cd1/ML/Pytorch/CNN_architectures/pytorch_vgg_implementation.py#L16-L62
    avgpool is missing from Ref#1.
    Ref#2: https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html
    """

    def __init__(self, input_nc, num_classes, VGG_type):
        super(VGG_net, self).__init__()
        self.in_channels = input_nc
        self.conv_layers = self.create_conv_layers(VGG_types[VGG_type])
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1) #flatten
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        return nn.Sequential(*layers)


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [ nn.Sigmoid()] # changed to sigmoid from tanh

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

# Linear Kernel that learns.
class LinearKernel(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, dimension = 3):
        super(LinearKernel, self).__init__()
        _conv = conv(dimension)

        padding_size = np.round((kernel_size - 1) / 2)
        if dimension == 2:
            padding_size = (int(padding_size), int(padding_size))
        elif dimension ==3:
            padding_size = (int(padding_size), int(padding_size), int(padding_size))  # such padding size makes sure that the image stays as the same size.
        self.convlayer = _conv(input_nc, output_nc, kernel_size=kernel_size, stride=1, padding=padding_size, bias=False)

    def forward(self, inputs):
        outputs = self.convlayer(inputs)
        return outputs

class LinearKernel_double(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, dimension = 3):
        super(LinearKernel_double, self).__init__()
        _conv = conv(dimension)
        padding_size = np.round((kernel_size - 1) / 2)

        if dimension == 2:
            padding_size = (int(padding_size), int(padding_size))
        elif dimension ==3:
            padding_size = (int(padding_size), int(padding_size), int(padding_size))  # such padding size makes sure that the image stays as the same size.
        self.convlayer = _conv(input_nc, output_nc, kernel_size=kernel_size, stride=1, padding=padding_size, bias=False)

    def forward(self, inputs):
        hiddens = self.convlayer(inputs)
        outputs = self.convlayer(hiddens)
        return outputs

# Linear Kernel with noise channel
class LinearKernel_NC(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, dimension = 3):
        super(LinearKernel, self).__init__()
        _conv = conv(dimension)

        padding_size = np.round((kernel_size - 1) / 2)
        padding_size = (int(padding_size), int(padding_size), int(padding_size))  # such padding size makes sure that the image stays as the same size.
        self.blur_convlayer = _conv(input_nc, output_nc, kernel_size=kernel_size, stride=1, padding=padding_size, bias=False)
        self.noise_convlayer = _conv(input_nc, output_nc, kernel_size=kernel_size, stride=1, padding=padding_size, bias=False)

    def forward(self, inputs):
        outputs_1 = self.blur_convlayer(inputs)
        outputs_2 = self.noise_convlayer(inputs) #TODO: maybe add some regularizer for faster learning?
        outputs = outputs_1 + outputs_2

        return outputs

# Adopted Deep Linear Generator from Bell-Kligler et al.
#REF: http://www.wisdom.weizmann.ac.il/∼vision/kernelgan
class DeepLinearGenerator(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(DeepLinearGenerator, self).__init__()
        narrowing_kernels = [7,5,3]
        unit_kernels = [1,1]
        # hidden layers have 64 channels.
        self.first_layer = nn.Conv3d(in_channels=input_nc, out_channels=input_nc*64, kernel_size=narrowing_kernels[0], padding=3, bias=False)
        feature_block = [] # Stacking intermediate layer
        feature_block += [nn.Conv3d(in_channels=input_nc*64, out_channels=input_nc*64, kernel_size=narrowing_kernels[1], padding=2, bias=False)]
        feature_block += [nn.Conv3d(in_channels=input_nc*64, out_channels=input_nc*64, kernel_size=narrowing_kernels[2], padding=1, bias=False)]

        for layer in range(len(unit_kernels)):
            feature_block += [nn.Conv3d(in_channels=input_nc*int(64*((1/2)**layer)), out_channels=input_nc*int(64*((1/2)**(layer+1))), kernel_size=unit_kernels[layer], padding=0, bias=False)]

        self.feature_block = nn.Sequential(*feature_block)

        # Final layer
        # NOTE: different from KernelGAN, we do not apply downsampling here.
        self.final_layer = nn.Conv3d(in_channels=input_nc*int(64*((1/2)**(layer+1))), out_channels=output_nc, kernel_size=unit_kernels[-1], padding=0, bias=False)

    def forward(self, input):
        downscaled = self.first_layer(input)
        features = self.feature_block(downscaled)
        output = self.final_layer(features)
        return output


# Fixed linear Kernel
class FixedLinearKernel(nn.Module):
    def __init__(self, psf, noise_setting):
        super(FixedLinearKernel, self).__init__()
        self.psf = nn.Parameter(psf, requires_grad = False)
        # self.psf = self.psf.unsqueeze_(dim=0) # adds batch channel.
        # self.psf = self.psf.unsqueeze_(dim=0) # adds color channel.
        self.kernel_size = np.asarray(self.psf.shape[2:])
        self.gau_sigma, self.poisson_peak = noise_setting

    def forward(self, input):
        padding_size = np.round((self.kernel_size - 1) / 2)
        padding_size = tuple(padding_size.astype(int))  # such padding size makes sure that the image stays as the same size.
        convd = F.conv3d(input, self.psf, stride=1, padding=padding_size)
        if self.kernel_size[-1] % 2 == 0:  # If kernel_size is a even number, it should be center-cropped by one.
            convd = convd[:, :, 1:, 1:, 1:]
        convd_noised = noisy('gauss', convd, sigma = self.gau_sigma, is_tensor=True)
        convd_noised = noisy('poisson', convd_noised, peak = self.poisson_peak, is_tensor=True)

        return convd_noised


class FiLM(nn.Module):
    def __init__(self, input_nc):
        super(FiLM, self).__init__()
        narrowing_kernels = [5, 3]
        unit_kernels = [1, 1]

        # hidden layers have 64 channels.
        feature_block = []  # Stacking intermediate layer
        feature_block += [
            nn.Conv3d(in_channels=input_nc, out_channels=input_nc * 64, kernel_size=narrowing_kernels[0],
                      padding=2, bias=False)]
        feature_block += [
            nn.Conv3d(in_channels=input_nc * 64, out_channels=input_nc * 64, kernel_size=narrowing_kernels[1],
                      padding=1, bias=False)]

        for layer in range(len(unit_kernels)):
            feature_block += [nn.Conv3d(in_channels=input_nc * int(64 * ((1 / 2) ** layer)),
                                        out_channels=input_nc * int(64 * ((1 / 2) ** (layer + 1))),
                                        kernel_size=unit_kernels[layer], padding=0, bias=False)]

        self.feature_block = nn.Sequential(*feature_block)

        # Final layer
        # NOTE: different from KernelGAN, we do not apply downsampling here.
        #TODO Figure out the output channel
        self.final_layer = nn.Linear(in_features=input_nc*int(64*((1/2)**(layer+1))), out_features=2)

    def forward(self, input):
        features = self.feature_block(input)
        output = self.final_layer(features)
        return output

#### OLD UNET IMPLEMENTATION FOR SOMA SEGMENTATION ####
class UnetTwoOuts(nn.Module):

    def __init__(self, input_nc, output_nc):

        super(UnetTwoOuts, self).__init__()
        # Downsampling
        self.double_conv1 = double_conv(1,  input_nc, 3, 1, 1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        self.double_conv2 = double_conv(input_nc, input_nc*2, 3, 1, 1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        self.double_conv3 = double_conv(input_nc * 2, input_nc*4, 3, 1, 1)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)

        # bottom floor
        self.double_conv5 = double_conv(input_nc *4, input_nc*8, 3, 1, 1)

        # Upsampling = transposed convolution
        self.t_conv3 = nn.ConvTranspose3d(input_nc*8, input_nc*4, 2, 2)
        self.ex_double_conv3 = double_conv(input_nc*8, input_nc*4, 3, 1, 1)

        self.t_conv2 = nn.ConvTranspose3d(input_nc*4, input_nc*2, 2, 2)
        self.ex_double_conv2 = double_conv(input_nc*4, input_nc*2, 3, 1, 1)

        self.t_conv1 = nn.ConvTranspose3d(input_nc*2, input_nc, 2, 2)
        self.ex_double_conv1 = double_conv(input_nc*2, input_nc, 3, 1, 1)

        # last stage
        self.one_by_one = nn.Conv3d(input_nc, output_nc, 1, 1, 0)
        self.one_by_one_2 = double_conv(input_nc, 1, 1, 1, 0)

    def forward(self, inputs):

        # Contracting Path
        conv1 = self.double_conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.double_conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.double_conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        # bottom floor
        conv5 = self.double_conv5(maxpool3)

        t_conv3 = self.t_conv3(conv5)
        cat3 = torch.cat([conv3, t_conv3], 1)
        ex_conv3 = self.ex_double_conv3(cat3)

        t_conv2 = self.t_conv2(ex_conv3)
        cat2 = torch.cat([conv2, t_conv2], 1)
        ex_conv2 = self.ex_double_conv2(cat2)

        t_conv1 = self.t_conv1(ex_conv2)
        cat1 = torch.cat([conv1, t_conv1], 1)
        ex_conv1 = self.ex_double_conv1(cat1)

        one_by_one = self.one_by_one(ex_conv1)
        one_by_one_2 = self.one_by_one_2(ex_conv1)

        return (one_by_one, one_by_one_2)


######################END OF PETER'S IMPLEMENTATION##############################################
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=None, use_sigmoid = False, dimension =3):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """

        _conv = conv(dimension)

        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm3d has affine parameters
            use_bias = norm_layer.func == instance_norm(dimension)
        else:
            use_bias = norm_layer == instance_norm(dimension)

        kw = 4
        padw = 1
        sequence = [_conv(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)

            sequence += [
                _conv(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        sequence += [
            _conv(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [_conv(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map

        if use_sigmoid:
            print ("Using sigmoid in the last layer of Discriminator. Note that LSGAN may work well with this loss.")
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        # is_cuda = next(self.model.parameters()).is_cuda
        return self.model(input)

class NLayerDiscriminatorSN(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer = None, use_sigmoid=False, dimension=3):
        super(NLayerDiscriminatorSN, self).__init__()

        _conv = conv(dimension)
        use_bias = False

        kw = 4
        padw = 1
        sequence = [
            nn.utils.spectral_norm(_conv(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.utils.spectral_norm(_conv(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.utils.spectral_norm(_conv(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.utils.spectral_norm(_conv(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class KernelPatchDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator
    We adopt the patchGAN model from the KernelGAN.
    """

    def __init__(self, input_nc, ndf=64, n_layers=5, norm_layer=None, dimension =3):
        super(KernelPatchDiscriminator, self).__init__()

        _conv = conv(dimension)

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm3d has affine parameters
            use_bias = norm_layer.func == instance_norm(dimension)
        else:
            use_bias = norm_layer == instance_norm(dimension)


        # First layer - Convolution (with no ReLU)
        self.first_layer = _conv(in_channels=input_nc, out_channels=ndf, kernel_size=7, bias=use_bias)

        feature_block = []  # Stacking layers with 1x1 kernels
        for _ in range(1, n_layers-1): # not including the final layer
            feature_block += [_conv(in_channels=ndf, out_channels=ndf, kernel_size=1, bias=use_bias),
                              norm_layer(ndf),
                              nn.ReLU(True)]

        self.feature_block = nn.Sequential(*feature_block)

        self.final_layer = _conv(in_channels=ndf, out_channels=1, kernel_size=1, bias=use_bias)

    def forward(self, input_tensor):
        receptive_extraction = self.first_layer(input_tensor)
        features = self.feature_block(receptive_extraction)
        return self.final_layer(features)

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=batch_norm, dimension = 3):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """

        _conv = conv(dimension)

        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm3d has affine parameters
            use_bias = norm_layer.func == instance_norm(dimension)
        else:
            use_bias = norm_layer == instance_norm(dimension)

        self.net = [
            _conv(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            _conv(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            _conv(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, dimension=3, num_downs=5, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, op_dim=dimension, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, op_dim=dimension, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, op_dim=dimension, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, op_dim=dimension, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, op_dim=dimension, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, op_dim=dimension, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, op_dim=3, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == instance_norm
        else:
            use_bias = norm_layer == instance_norm
            
        if input_nc is None:
            input_nc = outer_nc

        downconv_ = conv(op_dim)
        transconv_ = convtranspose(op_dim)

        downconv = downconv_(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        
         
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = transconv_(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = transconv_(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = transconv_(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)