import torch
import torch.nn as nn

from . import util
import layer

# Model: conv 3x3,5 -> relu -> conv 3x3,10 -> relu -> fc 10
# Shape: 1x10x10 -> 5x8x8  -> 10x6x6 -> 10

ishape = (1, 10, 10)
Poc1Model = nn.Sequential(
    nn.Conv2d(1, 5, 3),
    nn.ReLU(),
    nn.Conv2d(5, 10, 3),
    nn.ReLU(),
    nn.Linear(360, 10)
)

class Poc1Client():
    def __init__(self, socket) -> None:
        self.socket = socket
        self.layers = util.make_client_model(socket, Poc1Model, ishape)
        
    def offline(self):
        for lyr in self.layers:
            lyr.offline()
    
    def online(self, x):
        for lyr in self.layers:
            x = lyr.online(x)
        return x
    
class Poc1Server():
    def __init__(self, socket) -> None:
        self.socket = socket
        self.layers = util.make_server_model(socket, Poc1Model, ishape)
    
    def offline(self):
        for lyr in self.layers:
            lyr.offline()
            
    def online(self, x):
        for lyr in self.layers:
            x = lyr.online(x)
        return x
    