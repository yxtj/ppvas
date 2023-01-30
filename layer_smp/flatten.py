from .base import LocalLayerClient, LocalLayerServer

from socket import socket
from typing import Union
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
        
    def setup(self, last_lyr: LocalLayerServer, m: Union[torch.Tensor, float, int]=None, **kwargs) -> None:
        t = time.time()
        m = self.layer(last_lyr.m)
        super().setup(last_lyr, m)
        self.stat.time_offline += time.time() - t
        
    