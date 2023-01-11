from layer.base import LocalLayerClient, LocalLayerServer
# from comm.util import send_torch, recv_torch

from socket import socket
import torch
import torch.nn as nn

class ReLUClient(LocalLayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple) -> None:
        super().__init__(socket, ishape, oshape)
        self.layer = nn.ReLU()
    
    def online(self, xm) -> torch.Tensor:
        return self.layer(xm)
    
class ReLUServer(LocalLayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple,
                 layer: torch.nn.Module, m_last: torch.Tensor) -> None:
        assert isinstance(layer, nn.ReLU)
        super().__init__(socket, ishape, oshape, layer, m_last)
