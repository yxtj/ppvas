import socket
import torch
import torch.nn as nn
from Pyfhel import Pyfhel

from model import util
# import layer

class Client():
    def __init__(self, socket: socket.socket, model: nn.Module, inshape: tuple, he: Pyfhel):
        self.socket = socket
        # model
        self.model = model
        self.inshape = inshape
        # he
        self.he = he
        # layers
        self.layers, self.shortcuts = util.make_client_model(socket, model, inshape, he)
        self.to_buffer = [v for k,v in self.shortcuts.items()]
        
    def offline(self):
        for i, lyr in enumerate(self.layers):
            print('  offline {}: {} ...'.format(i, lyr.__class__.__name__))
            # setup
            if i in self.shortcuts:
                idx = self.shortcuts[i]
                r_other = self.layers[idx].r
                lyr.setup(r_other)
            # offline
            lyr.offline()
            
    def online(self, data: torch.Tensor) -> torch.Tensor:
        for lyr in self.layers:
            data = lyr.online(data)
        return data

