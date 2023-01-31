from .base import LayerClient, LayerServer

from socket import socket
from typing import Union
import time
import torch
import torch.nn as nn
from Pyfhel import Pyfhel

class ConvClient(LayerClient):
    def __init__(self, socket:socket, ishape:tuple, oshape:tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)

    
class ConvServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, nn.Conv2d)
        super().__init__(socket, ishape, oshape, layer)
    
    def setup(self, last_lyr: LayerServer, m: Union[torch.Tensor, float, int]=None, **kwargs) -> None:
        t = time.time()
        super().setup(last_lyr, m)
        self.stat.time_offline += time.time() - t
    
    def offline(self) -> torch.Tensor:
        t = time.time()
        data = self.recv_he()
        rm = self.reconstruct_mul_data(data) # r_i / m_{i-1}
        data = self.run_layer_offline(rm) # W_i * r_i / m_{i-1}
        data = self.construct_mul_share(data) # W_i * r_i / m_{i-1} .* m_{i}
        self.send_he(data)
        self.stat.time_offline += time.time() - t
        return rm
        
    def online(self) -> torch.Tensor:
        t = time.time()
        data = self.recv_plain() # xmr_i = x_i * m_{i-1} - r_i
        xrm = self.reconstruct_mul_data(data) # x_i - r_i / m_{i-1}
        data = self.layer(xrm) # W_i * (x_i - r_i / m_{i-1})
        data = self.construct_mul_share(data) # W_i * (x_i - r_i / m_{i-1}) .* m_{i}
        self.send_plain(data)
        self.stat.time_online += time.time() - t
        return xrm

