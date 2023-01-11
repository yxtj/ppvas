import socket
import torch
import torch.nn as nn

from model import util
import layer

class Client():
    def __init__(self, socket: socket.socket, model: nn.Module, inshape: tuple):
        self.socket = socket
        # model
        self.model = model
        self.inshape = inshape
        # layers
        self.layers = util.make_client_model(socket, model, inshape)
        
    def offline(self):
        for lyr in self.layers:
            lyr.offline()
            
    def online(self, data: torch.Tensor) -> torch.Tensor:
        for lyr in self.layers:
            data = lyr.online(data)
        return data

