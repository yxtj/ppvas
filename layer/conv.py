from layer.base import LayerClient, LayerServer

from socket import socket
import torch
import torch.nn as nn
from Pyfhel import Pyfhel

class Conv2DClient(LayerClient):
    def __init__(self, socket:socket, ishape:tuple, oshape:tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)

    
class Conv2DServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple,
                 layer: torch.nn.Module, m_last: torch.Tensor) -> None:
        assert isinstance(layer, nn.Conv2d)
        super().__init__(socket, ishape, oshape, layer, m_last)
        self.set_m_positive()
        # TODO: construct HE layer
        
    def offline(self) -> torch.Tensor:
        # r_i = recv_torch(self.socket)
        r_i = self.recv_he()
        data = self.reconstruct_mul_data(r_i) # r_i / m_{i-1}
        data = self.layer(data) # W_i * r_i / m_{i-1}
        data = self.construct_mul_share(data) # w_i * r_i / m_{i-1} .* m_{i}
        self.send_he(data)
        return r_i
        
    def online(self) -> torch.Tensor:
        xmr_i = self.recv_plain() # xmr_i = x_i * m_{i-1} - r_i
        data = self.reconstruct_mul_data(xmr_i) # x_i - r_i / m_{i-1}
        data = self.layer(data) # W_i * (x_i - r_i / m_{i-1})
        data = self.construct_mul_share(data) # W_i * (x_i - r_i / m_{i-1}) .* m_{i}
        self.send_plain(data)
        return xmr_i

