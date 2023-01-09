from layer.base import LayerClient, LayerServer
from comm.util import send_torch, recv_torch

from socket import socket
import torch
import torch.nn as nn

class ReLUClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple) -> None:
        super().__init__(socket, ishape, oshape)
        self.layer = torch.nn.ReLU()
        
    def offline(self) -> None:
        pass
    
    def online(self, xm) -> torch.Tensor:
        return self.layer(xm)
    
class ReLUServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple,
                 layer: torch.nn.Module, m_last: torch.Tensor) -> None:
        assert isinstance(layer, nn.ReLU)
        super().__init__(socket, ishape, oshape, layer, m_last)
        
    def offline(self) -> None:
        pass
    
    def online(self) -> None:
        pass