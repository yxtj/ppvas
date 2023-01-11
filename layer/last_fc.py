from layer.base import LayerClient, LayerServer
from comm.util import send_torch, recv_torch

from socket import socket
import torch
import torch.nn as nn

class LastFcClient(LayerClient):
    def __init__(self, socket: socket, inshape: tuple, outshape: tuple) -> None:
        super().__init__(socket, inshape, outshape)
    
    
class LastFcServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple,
                 layer: torch.nn.Module, mlast: torch.Tensor) -> None:
        assert isinstance(layer, nn.Linear)
        super().__init__(socket, ishape, oshape, layer, mlast)
        self.m = None
    
    def offline(self) -> torch.Tensor:
        r_i = recv_torch(self.socket) # r_i
        data = self.reconstruct_mul_data(r_i) # r_i / m_{i-1}
        data = self.layer(data) # W_i * r_i / m_{i-1}
        send_torch(self.socket, data)
        return r_i
    
    def online(self) -> torch.Tensor:
        xmr_i = recv_torch(self.socket) # xmr_i = x_i * m_{i-1} - r_i
        data = self.reconstruct_mul_data(xmr_i) # x_i - r_i / m_{i-1}
        data = self.layer(data) # W_i * (x_i - r_i / m_{i-1})
        send_torch(self.socket, data)
        return xmr_i
    