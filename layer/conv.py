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
        
    def offline(self) -> torch.Tensor:
        r_i = recv_torch(self.socket)
        # TODO: use HE layer
        data = self.reconstruct_mul_data(r_i) # r_i / m_{i-1}
        data = self.layer(data) # W_i * r_i / m_{i-1}
        data = self.construct_mul_share(data) # w_i * r_i / m_{i-1} .* m_{i}
        send_torch(self.socket, data)
        return r_i
        
    def online(self) -> torch.Tensor:
        xmr_i = recv_torch(self.socket) # xmr_i = x_i * m_{i-1} - r_i
        data = self.reconstruct_mul_data(xmr_i) # x_i - r_i / m_{i-1}
        data = self.layer(data) # W_i * (x_i - r_i / m_{i-1})
        data = self.construct_mul_share(data) # W_i * (x_i - r_i / m_{i-1}) .* m_{i}
        send_torch(self.socket, data)
        return xmr_i

