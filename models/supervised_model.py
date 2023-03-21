import torch
from .base_model import BaseModel
from torchmetrics import StructuralSimilarityIndexMeasure
from . import networks


class SupervisedModel(BaseModel):
    """
    This model is a supervised model for comparison.

    """
    def __init__(self, opt):
        """Initialize the pix2pix class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)

        if opt.data_gt is not None:
            self.validate = True
        else:
            self.validate = False

        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1']
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real', 'fake']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G' + opt.model_suffix]  # only generator is needed.
        self.dimension = opt.image_dimension

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, dimension = self.dimension)



        if self.isTrain:
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

            if self.validate:
                self.criterionValL1 = torch.nn.L1Loss() # comparison with GT for validation
                self.criterionValssim =StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

        self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_src = input['src' if AtoB else 'tgt'].to(self.device)
        self.real_tgt = input['tgt' if AtoB else 'src'].to(self.device)

        if self.validate:
            self.real_gt = input['gt'].to(self.device)
            self.image_paths_gt = input['gt_paths']

        self.image_paths_src = input['src_paths' if AtoB else 'tgt_paths']
        self.image_paths_tgt = input['tgt_paths' if AtoB else 'src_paths']

        self.cube_shape = self.real_src.shape
        self.num_slice = self.cube_shape[-3]

    def forward(self):
        """Run forward pass."""
        self.fake = self.netG(self.real_tgt)  # G(real)
        self.fake_val = self.netG(self.real_src)


    def backward_G(self):
        self.criterionL1(self.real, )

    def optimize_parameters(self):
        """No optimization for test model."""
        pass