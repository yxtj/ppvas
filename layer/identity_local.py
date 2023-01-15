from layer.base import LocalLayerClient, LocalLayerServer

from socket import socket
import time
import torch
import torch.nn as nn
from Pyfhel import Pyfhel

class LocalIdentityClient(LocalLayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)
        self.layer = nn.ReLU()
    
    def online(self, xm) -> torch.Tensor:
        return xm
    
class LocalIdentityServer(LocalLayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, nn.Identity)
        super().__init__(socket, ishape, oshape, layer)

    def setup(self, mlast:torch.Tensor) -> None:
        super().setup(mlast)
        self.m = mlast
        