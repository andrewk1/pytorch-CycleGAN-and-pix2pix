import torch
import copy
from itertools import chain
from util.image_pool import ImagePool
from .base_model import BaseModel
from .canon_to_pi_model import CanonToPiModel
from . import networks


class ZyDiscrimModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

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

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        #parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['ZY', 'ZY_fake_noisy', 'ZY_fake_sampled']
        self.visual_names = ['canonical']

        self.d_update = 0

        self.model_names = ['ZY']  # G_sem', 
        self.netG = networks.define_G(opt.input_nc, 5, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        optPI = copy.deepcopy(opt)
        optPI.name = "canon3pi"
        optPI.continue_train = True
        self.netPredPI = CanonToPiModel(optPI)
        self.netPredPI.setup(optPI)
        self.netPredPI = self.netPredPI.netPI
        self.netPredPI.eval()

        # This network is for predicting whether (canonical, predicted PI) is true
        self.netZY = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                       opt.n_layers_D, opt.norm, opt.init_type, 
                                       opt.init_gain, self.gpu_ids, pi=True)

        # TODO: Figure out if I should keep this image buffer
        self.fake_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

        # define loss functions
        self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.

        self.optimizer_ZY = torch.optim.Adam(self.netZY.parameters(),
                                             lr=opt.lr,
                                             betas=(opt.beta1, 0.999))

        self.optimizers.append(self.optimizer_ZY)

        # TODO: Tune the variance
        self.m = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([0.1]))

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.canonical = input['canonical'].to(self.device)
        self.sampled_canonical = input['sampled_canonical'].to(self.device)

    def backward_ZY(self):
        pred_real = self.netZY(self.canonical, self.netPredPI(self.canonical))
        self.loss_ZY_real = self.criterionGAN(pred_real, True)

        noisy_pi_pred = self.netPredPI(self.canonical + self.m.sample(self.canonical.shape).view((self.opt.batch_size, 3, 256, 256)).to(self.device))
        pred_fake_noisy = self.netZY(self.canonical, noisy_pi_pred)
        pred_fake_sampled = self.netZY(self.sampled_canonical, self.netPredPI(self.canonical).to(self.device))

        self.loss_ZY_fake_noisy = self.criterionGAN(pred_fake_noisy, False) 
        self.loss_ZY_fake_sampled = self.criterionGAN(pred_fake_sampled, False)

        self.loss_ZY = (self.loss_ZY_real + self.loss_ZY_fake_noisy + self.loss_ZY_fake_sampled) * (1. / 3)
        self.loss_ZY.backward()

    def forward(self):
        pass

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.optimizer_ZY.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_ZY()      # calculate gradients for D_A
        self.optimizer_ZY.step()
