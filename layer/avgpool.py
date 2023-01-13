from layer.base import LayerClient, LayerServer

from socket import socket
import torch
import torch.nn as nn
from Pyfhel import Pyfhel

class AvgPoolClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)
        self.layer = None
        
    def setup(self, layer:torch.nn.AvgPool2d):
        self.layer = layer
        
    def online(self, xm) -> torch.Tensor:
        self.send_he(xm)
        data = self.recv_he()
        data = self.reconstruct_add_data(data)
        return data

class AvgPoolServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple,
                 layer: torch.nn.Module, m_last: torch.Tensor) -> None:
        assert isinstance(layer, nn.AvgPool2d)
        super().__init__(socket, ishape, oshape, layer, m_last)
        # set m
        self.set_m_any()
        
    def offline(self) -> torch.Tensor:
        r_i = self.recv_he() # r_i
        data = self.reconstruct_mul_data(r_i) # r_i / m_{i-1}
        data = self.layer(data) # avg_pool( r_i / m_{i-1} )
        data = self.construct_mul_share(data, self.mp) # avg_pool( r_i / m_{i-1} ) .* m_{i}
        self.send_he(data)
        return r_i
        
    def online(self) -> torch.Tensor:
        xmr_i = self.recv_plain() # xmr_i = x_i * m_{i-1} - r_i
        data = self.reconstruct_mul_data(xmr_i) # x_i - r_i / m_{i-1}
        data = self.layer(data) # avg_pool( x_i - r_i / m_{i-1} )
        data = self.construct_mul_share(data, self.mp) # avg_pool( x_i - r_i / m_{i-1} ) .* m^p_{i}
        self.send_plain(data)
        return xmr_i
    