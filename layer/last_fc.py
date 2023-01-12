from layer.base import LayerClient, LayerServer

from socket import socket
import torch
import torch.nn as nn
from Pyfhel import Pyfhel

class LastFcClient(LayerClient):
    def __init__(self, socket: socket, inshape: tuple, outshape: tuple, he:Pyfhel) -> None:
        super().__init__(socket, inshape, outshape, he)
    
    
class LastFcServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple,
                 layer: torch.nn.Module, mlast: torch.Tensor) -> None:
        assert isinstance(layer, nn.Linear)
        super().__init__(socket, ishape, oshape, layer, mlast)
        self.m = None
    
    def offline(self) -> torch.Tensor:
        r_i = self.recv_he() # r_i
        data = self.reconstruct_mul_data(r_i) # r_i / m_{i-1}
        data = self.layer(data) # W_i * r_i / m_{i-1}
        self.send_he(data)
        return r_i
    
    def online(self) -> torch.Tensor:
        xmr_i = self.recv_plain() # xmr_i = x_i * m_{i-1} - r_i
        data = self.reconstruct_mul_data(xmr_i) # x_i - r_i / m_{i-1}
        data = self.layer(data) # W_i * (x_i - r_i / m_{i-1})
        self.send_plain(data)
        return xmr_i
    