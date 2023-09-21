from torch import nn
from torch.nn import Parameter
from torch.nn import Conv2d, ConvTranspose2d, Linear
import math
from utils import calculate_kl as KL

class ModuleProb(nn.Module):
    def __init__(self):
        super(ModuleProb, self).__init__()

    def forward(self, x):
        for module in self.children():
            x = module(x)

        kl = 0.0
        for module in self.module():
            if hasattr(module, 'kl_loss'):
                kl += module.kl_loss()

        return x, kl
    
class FlattenLayer(ModuleProb):
    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features
        
    def forward(self, x):
        return x.view(-1, self.num_features)
    
class LinearB(ModuleProb):
    def __init__(self, in_features, out_features, bias=True, priors=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu': 0.0,
                'prior_sigma': 1.0,
                'posterior_mu_initial': (0, 1),
                'posterior_rho_initial': (-3, 1)
            }

        self.W_mu = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.W_rho = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if self.bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_rho = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, x, sample=True):
        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.bias:
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_var = self.bias_sigma**2
        else:
            self.bias_sigma = None
            bias_var = None

        act_mu = Linear(x, self.W_mu, self.bias_mu)
        act_var = 1e-16 + Linear(x**2, self.W_sigma**2, bias_var)
        act_std = torch.sqrt(act_var)

        if self.training or sample:
            eps = torch.empty(act_mu.size()).normal_(0, 1).to(self.device)
            return act_mu + act_std * eps
        else:
            return act_mu
        
    def kl_loss(self):
        kl = KL(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.bias:
            kl += KL(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl

    
class Conv2dB(ModuleProb):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, priors=None):
        super(Conv2dB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride=stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if priors is None:
            priors = {
                'prior_mu': 0.0,
                'prior_sigma': 1.0,
                'posterior_mu_initial': (0, 1),
                'posterior_rho_initial': (-3, 1)
            }

        self.W_mu = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.W_rho = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if self.bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_rho = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, x, sample=True):
        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.bias:
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_var = self.bias_sigma**2
        else:
            self.bias_sigma = None
            bias_var = None

        act_mu = Conv2d(x, self.W_mu, self.bias_mu, self.stride, self.padding, self.dilation)
        act_var = 1e-16 + Conv2d(x**2, self.W_sigma**2, bias_var, self.stride, self.padding, self.dilation)
        act_std = torch.sqrt(act_var)

        if self.training or sample:
            eps = torch.empty(act_mu.size()).normal_(0, 1).to(self.device)
            return act_mu + act_std * eps
        else:
            return act_mu
        
    def kl_loss(self):
        kl = KL(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.bias:
            kl += KL(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl
    
class ConvTranspose2dB(ModuleProb):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, priors=None):
        super(ConvTranspose2dB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride=stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu': 0.0,
                'prior_sigma': 1.0,
                'posterior_mu_initial': (0, 1),
                'posterior_rho_initial': (-3, 1)
            }

        self.W_mu = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.W_rho = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if self.bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_rho = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, x):
        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.bias:
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_var = self.bias_sigma**2
        else:
            self.bias_sigma = None
            bias_var = None

        act_mu = ConvTranspose2d(x, self.W_mu, self.bias_mu, self.stride, self.padding, self.dilation)
        act_var = 1e-16 + ConvTranspose2d(x**2, self.W_sigma**2, bias_var, self.stride, self.padding, self.dilation)
        act_std = torch.sqrt(act_var)

        if self.training or sample:
            eps = torch.empty(act_mu.size()).normal_(0, 1).to(self.device)
            return act_mu + act_std * eps
        else:
            return act_mu
        
    def kl_loss(self):
        kl = KL(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.bias:
            kl += KL(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl

class ConvBlock(ModuleProb):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out