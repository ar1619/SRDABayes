import torch 
import torch.nn as nn
from network_elements import Conv2dB, LinearB, FlattenLayer, ModuleProb

class TestNet(ModuleProb):
    