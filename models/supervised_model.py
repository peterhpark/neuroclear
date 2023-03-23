import torch
from .base_model import BaseModel
from torchmetrics import StructuralSimilarityIndexMeasure
from . import networks


class SupervisedModel(BaseModel):
    """
    This model is a supervised model for comparison.
    """

    def __init__(self, opt):
        """Initialize the simple supervised learning class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseModel.__init__(self, opt)

        if opt.validate:
            self.validate = True
        else:
            self.validate = False

        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = ['L1']
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_src', 'fake', 'real_tgt']

        if self.validate:
            self.loss_names += ['valL1', 'valssim']
            self.visual_names += ['real_src_val', 'real_tgt_val']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']  # only generator is needed.

        self.dimension = opt.image_dimension

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, dimension = self.dimension)

        if self.isTrain:
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

            if self.validate:
                self.criterionValL1 = torch.nn.L1Loss() # comparison with GT for validation
                self.criterionValssim =StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

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
            self.real_src_val = input['src_val'].to(self.device)
            self.real_tgt_val = input['tgt_val'].to(self.device)

        self.image_paths_src = input['src_paths' if AtoB else 'tgt_paths']
        self.image_paths_tgt = input['tgt_paths' if AtoB else 'src_paths']

        self.cube_shape = self.real_src.shape
        self.num_slice = self.cube_shape[-3]

    def forward(self):
        """Run forward pass."""
        self.fake = self.netG(self.real_src)  # G(real)
        if self.validate:
            self.fake_val = self.netG(self.real_src_val)

    def backward_G(self):
        self.loss_L1 = self.criterionL1(self.real_tgt, self.fake)

        if self.validate:
            self.loss_valL1 = self.criterionValL1(self.real_tgt_val, self.fake_val.detach())
            self.loss_valssim = self.criterionValssim(self.real_tgt_val, self.fake_val.detach())
        self.loss_L1.backward()

    def optimize_parameters(self):
        self.forward()
        self.backward_G()
        self.optimizer_G.step()
