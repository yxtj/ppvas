import torch
import torch.nn as nn

class ShortCut(nn.Module):
    def __init__(self, otherlayer:int) -> None:
        '''
        otherlayer: the relative index of the other layer in the model
        '''
        super().__init__()
        self.otherlayer = otherlayer
        
    def forward(self, x, y):
        return x+y
    