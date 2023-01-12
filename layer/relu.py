from layer.base import LocalLayerClient, LocalLayerServer

from socket import socket
import torch
import torch.nn as nn
from Pyfhel import Pyfhel

class ReLUClient(LocalLayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)
        self.layer = nn.ReLU()
    
    def online(self, xm) -> torch.Tensor:
        return self.layer(xm)
    
class ReLUServer(LocalLayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple,
                 layer: torch.nn.Module, m_last: torch.Tensor) -> None:
        assert isinstance(layer, nn.ReLU)
        super().__init__(socket, ishape, oshape, layer, m_last)
