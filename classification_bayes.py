import torch 
import torch.nn as nn
from network_elements import Conv2dB, LinearB, FlattenLayer, ModuleProb

class TestNet(ModuleProb):
    def __init__(self, inputs, outputs, priors, activation_type = 'softplus'):
        super(TestNet, self).__init__()

        