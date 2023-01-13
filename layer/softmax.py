from layer.base import LocalLayerClient, LocalLayerServer

from socket import socket
import time
import torch
import torch.nn as nn
from Pyfhel import Pyfhel

class SoftmaxClient(LocalLayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)
        self.layer = nn.Softmax()
    
    def online(self, xm) -> torch.Tensor:
        t = time.time()
        data = self.layer(xm)
        self.stat.time_online = time.time() - t
        return data
    

class SoftmaxServer(LocalLayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple,
                 layer: torch.nn.Module, mlast: torch.Tensor) -> None:
        assert isinstance(layer, nn.Softmax)
        super().__init__(socket, ishape, oshape, layer, mlast)
        
