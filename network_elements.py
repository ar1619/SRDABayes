from torch import nn
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
        self.kernel_size = kernel_size
        self.stride=stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        if priors is None:
            priors = {
                'prior_mu': 0.0,
                'prior_sigma': 1.0,
            }

    def forward(self, x):
        return self.conv(x)