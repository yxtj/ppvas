from .base import LocalLayerClient, LocalLayerServer

from socket import socket
import time
import torch
import torch.nn as nn
from Pyfhel import Pyfhel

class FlattenClient(LocalLayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)
        self.layer = torch.nn.Flatten()
    
    def online(self, xm) -> torch.Tensor:
        t = time.time()
        data = self.layer(xm)
        self.stat.time_online += time.time() - t
        return data


class FlattenServer(LocalLayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, nn.Flatten)
        super().__init__(socket, ishape, oshape, layer)
        
    def setup(self, mlast: torch.Tensor, m_other:torch.Tensor=None, identity_m:bool=False) -> None:
        super().setup(mlast, m_other=m_other, identity_m=identity_m)
        self.m = self.layer(mlast)
        
    