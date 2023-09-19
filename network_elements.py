from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math

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
    
class Conv2DB(ModuleProb):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, priors=None):
        super(Conv2DB, self).__init__()
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

        act_mu = F.conv2d(x, self.W_mu, self.bias_mu, self.stride, self.padding, self.dilation)
        act_var = 1e-16 + F.conv2d(x**2, self.W_sigma**2, bias_var, self.stride, self.padding, self.dilation)
        act_std = torch.sqrt(act_var)

        if self.training or sample:
            eps = torch.empty(act_mu.size()).normal_(0, 1).to(self.device)
            return act_mu + act_std * eps
        else:
            return act_mu