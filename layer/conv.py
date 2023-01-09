from layer.base import LayerClient, LayerServer
from comm.util import send_torch, recv_torch

from socket import socket
import torch
import torch.nn as nn

class Conv2DClient(LayerClient):
    def __init__(self, socket:socket, ishape:tuple, oshape:tuple) -> None:
        super().__init__(socket, ishape, oshape)

    
class Conv2DServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple,
                 layer: torch.nn.Module, m_last: torch.Tensor) -> None:
        assert isinstance(layer, nn.Conv2d)
        super().__init__(socket, ishape, oshape, layer, m_last)
        self.set_m_positive()
        # TODO: construct HE layer
        
    def offline(self) -> None:
        data = recv_torch(self.socket)
        # TODO: use HE layer
        data = self.layer(data)
        data = self.construct_mul_share(data)
        send_torch(self.socket, data)
        
    def online(self) -> None:
        data = recv_torch(self.socket)
        data = self.reconstruct_mul_data(data)
        data = self.layer(data)
        data = self.construct_mul_share(data)
        send_torch(self.socket, data)

