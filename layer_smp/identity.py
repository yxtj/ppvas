from .base import LayerClient, LayerServer

from socket import socket
from typing import Union
import time
import torch
import torch.nn as nn
from Pyfhel import Pyfhel

class IdentityClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)
        self.layer = nn.ReLU()
    

class IdentityServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, nn.Identity)
        super().__init__(socket, ishape, oshape, layer)
    
    def setup(self, last_lyr: LayerServer, m: Union[torch.Tensor, float, int]=None, **kwargs) -> None:
        t = time.time()
        super().setup(last_lyr, m)
        self.stat.time_offline += time.time() - t
    
    def offline(self) -> torch.Tensor:
        t = time.time()
        data = self.recv_he() # r_i
        rm = self.reconstruct_mul_data(data) # r_i / m_{i-1}
        data = self.construct_mul_share(rm) # r_i / m_{i-1} .* m_{i}
        self.send_he(data)
        self.stat.time_offline += time.time() - t
        return rm
    
    def online(self) -> torch.Tensor:
        t = time.time()
        data = self.recv_plain() # xmr_i = x_i * m_{i-1} - r_i
        xrm = self.reconstruct_mul_data(data) # x_i - r_i / m_{i-1}
        data = self.construct_mul_share(xrm) # (x_i - r_i / m_{i-1}) .* m_{i}
        self.send_plain(data)
        self.stat.time_online += time.time() - t
        return xrm
