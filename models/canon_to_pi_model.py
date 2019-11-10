
class CanonToPiModel(BaseModel):

	@staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
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
        self.loss_names = ['PI']  # 'sem',
        self.model_names = ['PI']
        self.netPI = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                       opt.n_layers_D, opt.norm, opt.init_type, 
                                       opt.init_gain, self.gpu_ids, fc=True)

        self.criterionPI = torch.nn.MSELoss()
        self.optimizer_PI = torch.optim.Adam(self.netPI.parameters(),
                                            lr=opt.lr,
                                            betas=(opt.beta1, 0.999))

        self.optimizers.append(self.optimizer_PI)

    def set_input(self, input):
        self.canonical = input['canonical'].to(self.device)
        self.canonical_state = input['canonical_pi'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.pred_state = self.netPI(self.canonical)

    def backward_PI(self):
        """Calculate the loss for generators G_canonical and G_pred"""
        self.loss_pi_real = self.criterionPI(self.pred_state, self.canonical_state) 
        self.loss_pi_real.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()      		  # compute fake images and reconstruction images.
        self.optimizer_PI.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_PI()             # calculate gradients for G_A and G_B
        self.optimizer_PI.step()       # update G_A and G_B's weights
