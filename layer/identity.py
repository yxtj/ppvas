from .base import LayerClient, LayerServer

from socket import socket
from typing import Union
import time
import torch
from Pyfhel import Pyfhel

class IdentityClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)
    

class IdentityServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, torch.nn.Identity)
        super().__init__(socket, ishape, oshape, layer)
    
    def offline(self) -> torch.Tensor:
        t = time.time()
        rm = self.recv_offline() # r_i * m_{i-1}
        self.send_offline(rm)
        self.stat.time_offline += time.time() - t
        return rm
    
    def online(self) -> torch.Tensor:
        t = time.time()
        xrm = self.recv_online() # x_i - r_i * m_{i-1}
        self.send_online(xrm)
        self.stat.time_online += time.time() - t
        return xrm
