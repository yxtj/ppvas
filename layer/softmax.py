from layer.base import LayerClient, LayerServer
from comm.util import send_torch, recv_torch

from socket import socket
import torch
import torch.nn as nn

class SoftmaxClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple) -> None:
        super().__init__(socket, ishape, oshape)
    
    def online(self, xm) -> torch.Tensor:
        data = self.construct_add_share(xm)
        send_torch(self.socket, data) # softmax(x_i) softmax(- r_i / m_{i-1})
        data = data / self.pre
        return data
    

class SoftmaxServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple,
                 layer: torch.nn.Module, mlast: torch.Tensor) -> None:
        assert isinstance(layer, nn.Softmax)
        super().__init__(socket, ishape, oshape, layer, mlast)
        
    def offline(self) -> None:
        data = recv_torch(self.socket) # r_i
        data = self.reconstruct_mul_data(data) # r_i / m_{i-1}
        data = self.layer(torch.neg(data)) # softmax(- r_i / m_{i-1})
        send_torch(self.socket, data)
    
    def online(self) -> None:
        data = recv_torch(self.socket) # x_i m_{i-1} - r_i
        data = self.reconstruct_mul_data(data) # x_i - r_i / m_{i-1}
        data = self.layer(data) # softmax(x_i - r_i / m_{i-1}) = softmax(x_i) softmax(- r_i / m_{i-1})
        send_torch(data)
        
        
