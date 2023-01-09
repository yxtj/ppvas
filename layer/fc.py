from layer.base import LayerClient, LayerServer
from comm.util import send_torch, recv_torch

from socket import socket
import torch
import torch.nn as nn

class FcClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple) -> None:
        super().__init__(socket, ishape, oshape)

class FcServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple,
                 layer: torch.nn.Module, m_last: torch.Tensor) -> None:
        assert isinstance(layer, nn.Linear)
        super().__init__(socket, ishape, oshape, layer, m_last)
        self.set_m_positive()
        
    def offline(self) -> None:
        data = recv_torch(self.socket)
        data = self.layer(data)
        data = self.construct_mul_share(data)
        send_torch(self.socket, data)
    
    def online(self) -> None:
        data = recv_torch(self.socket)
        data = self.reconstruct_mul_data(data)
        data = self.layer(data)
        data = self.construct_mul_share(data)
        send_torch(self.socket, data)
        
                   