import torch
from itertools import chain
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class PairedDiscrimRCANModel(BaseModel):
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
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # TODO: Maybe add idt_A and pi_A as extensions
        self.loss_names = ['pixel', 'seg', 'depth', 'discrim', 'G', 'D', 'pi_real', 'discrim_real']  # 'sem',
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real', 'random', 'canonical_pred_random', 'canonical', 'seg_pred_random', 'seg', 'depth', 'depth_pred_random', 'real', 'canonical_pred_real']

        self.d_update = 0
        self.beta = 0

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G', 'D']  # G_sem', 
        else:  # during test time, only load Gs
            self.model_names = ['G']


        self.netG = networks.define_G(opt.input_nc, 5, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators

            # Discriminator now takes 2 input images:
            # One image from pred/real canonical domain, one from input:
            # Input can be either random or real domains 
            # First 3 channels are input, last 3 are canonical
            # Pairs are: 
            # (self.random, self.canonical) Real 
            # (self.random, self.canonical_pred_random) Fake
            # (self.real, self.canonical_pred_real) Fake

            # Input either self.random or self.real
            # Output either self.canonical, self.canonical_pred_random, self.canonical_pred_real

            self.netD = networks.define_D(opt.output_nc * 2, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, 
                                          opt.init_gain, self.gpu_ids)

            self.netPI = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, 
                                          opt.init_gain, self.gpu_ids, fc=True)

            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)

            # TODO: Figure out if I should keep this image buffer
            self.random_sampled_canonical_pool = ImagePool(opt.pool_size) 
            self.random_canonical_pred_pool = ImagePool(opt.pool_size)
            self.canonical_pred_pool = ImagePool(opt.pool_size) 

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCanonical = torch.nn.MSELoss()              
            self.criterionSegmentation = torch.nn.MSELoss()
            self.criterionDepth = torch.nn.MSELoss()
            self.criterionPI = torch.nn.MSELoss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            #self.optimizer_G = torch.optim.Adam(self.netG_canonical.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(chain(self.netG.parameters(), self.netPI.parameters()),
                                                lr=opt.lr,
                                                betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr,
                                                betas=(opt.beta1, 0.999))

            #self.optimizer_PI = torch.optim.Adam(self.netPI.parameters(),
            #                                     lr=opt.lr,
            #                                     betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            #self.optimizers.append(self.optimizer_PI)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.canonical = input['canonical'].to(self.device)
        self.random = input['random'].to(self.device)
        self.seg = input['seg'].to(self.device)
        self.depth = input['depth'].to(self.device)
        self.real = input['real'].to(self.device)
        self.real_state = input['real_state'].type(torch.FloatTensor).to(self.device)
        self.sampled_canonical = input['sampled_canonical'].to(self.device)

        self.image_paths = input['random_path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        pred_real = self.netG(self.real)
        pred_random = self.netG(self.random)

        self.canonical_pred_real = pred_real[:,:3]

        self.canonical_pred_random = pred_random[:,:3]
        self.seg_pred_random       = pred_random[:,3:4]
        self.depth_pred_random     = pred_random[:,4:]

        self.random_canonical         = torch.cat((self.random, self.canonical), 1)
        self.random_sampled_canonical = torch.cat((self.random, self.sampled_canonical), 1)
        self.random_canonical_pred    = torch.cat((self.random, self.canonical_pred_random), 1)
        self.real_canonical_pred      = torch.cat((self.real, self.canonical_pred_real), 1)


    def backward_D_basic(self, netD, 
                         random_canonical_pair, 
                         random_sampled_canonical_pair, 
                         random_canonical_pred_pair,
                         random_sampled_canonical_pred_pair):
        """
        Calculate GAN loss for the discriminator
        """

        # netD must only say yes to canonical + corresponding random samples

        pred_random_canonical              = netD(random_canonical_pair)
        pred_random_sampled_canonical      = netD(random_sampled_canonical_pair.detach())
        pred_random_canonical_pred         = netD(random_canonical_pred_pair.detach())
        pred_random_sampled_canonical_pred = netD(random_sampled_canonical_pred_pair.detach())

        loss_D_real         = self.criterionGAN(pred_random_canonical, True)
        loss_D_sampled      = self.criterionGAN(pred_random_sampled_canonical, False)
        loss_D_pred         = self.criterionGAN(pred_random_canonical_pred, False)
        loss_D_sampled_pred = self.criterionGAN(pred_random_sampled_canonical_pred, False)

        loss_D = (loss_D_real + loss_D_sampled + loss_D_pred + loss_D_sampled_pred) * (1 / 4)
        loss_D.backward()
        return loss_D

    def backward_D(self):
        """Calculate GAN loss for discriminator D_A"""

        # Pairs are: 
        # (self.random, self.canonical) Real 
        # (self.random, self.canonical_pred_random) Fake
        # (self.real, self.canonical_pred_real) Fake
        random_sampled_canonical_pair = self.random_sampled_canonical_pool.query(self.random_sampled_canonical)
        random_canonical_pred_pair    = self.random_canonical_pred_pool.query(self.random_canonical_pred)
        random_sampled_canonical_pred = torch.cat((self.random, self.canonical_pred_pool.query(self.canonical_pred_random)), 1)

        self.loss_D = self.backward_D_basic(self.netD, 
                                            self.random_canonical, 
                                            random_sampled_canonical_pair, 
                                            random_canonical_pred_pair,
                                            random_sampled_canonical_pred)

    def backward_G(self):
        """Calculate the loss for generators G_canonical and G_pred"""

        self.loss_discrim = self.criterionGAN(self.netD(self.random_canonical_pred), True)
        self.loss_pixel   = self.criterionCanonical(self.canonical_pred_random, self.canonical)
        self.loss_seg     = self.criterionSegmentation(self.seg_pred_random, self.seg)
        self.loss_depth   = self.criterionDepth(self.depth_pred_random, self.depth)

        self.loss_discrim_real = self.criterionGAN(self.netD(self.real_canonical_pred), True)
        self.loss_pi_real      = self.criterionPI(self.netPI(self.canonical_pred_real), self.real_state) 

        self.loss_G = self.loss_pixel + self.loss_seg + self.loss_pi_real + \
                      self.loss_depth + self.beta * self.loss_discrim_real + self.loss_discrim

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        #self.optimizer_PI.zero_grad()
        self.backward_G()             # calculate gradients for G_A and G_B
        #self.optimizer_PI.step()
        self.optimizer_G.step()       # update G_A and G_B's weights

        # Only compute discrim loss when a canonical image exists (not in real domain)
        # D_A and D_B
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()      # calculate gradients for D_A
        if self.d_update % 5 == 0:
            self.optimizer_D.step()  # update D_A and D_B's weights
        self.d_update += 1
        self.beta = np.tanh(d_update)

