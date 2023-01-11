from layer.base import LocalLayerClient, LocalLayerServer
# from comm.util import send_torch, recv_torch

from socket import socket
import torch
import torch.nn as nn

class SoftmaxClient(LocalLayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple) -> None:
        super().__init__(socket, ishape, oshape)
        self.layer = nn.Softmax()
    
    def online(self, xm) -> torch.Tensor:
        data = self.layer(xm)
        return data
    

class SoftmaxServer(LocalLayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple,
                 layer: torch.nn.Module, mlast: torch.Tensor) -> None:
        assert isinstance(layer, nn.Softmax)
        super().__init__(socket, ishape, oshape, layer, mlast)
        
