from typing import Union
from .base import LayerClient, LayerServer

from socket import socket
from typing import Union
import time
import torch
import torch.nn as nn
from Pyfhel import Pyfhel

class AvgPoolClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel,
                 layer:torch.nn.AvgPool2d) -> None:
        super().__init__(socket, ishape, oshape, he)
        self.layer = layer
        

class AvgPoolServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, nn.AvgPool2d)
        super().__init__(socket, ishape, oshape, layer)
    
    def setup(self, last_lyr: LayerServer, m: Union[torch.Tensor, float, int]=None, **kwargs) -> None:
        t = time.time()
        super().setup(last_lyr, m)
        self.stat.time_offline += time.time() - t
        
    def offline(self) -> torch.Tensor:
        t = time.time()
        data = self.recv_he() # r_i
        rm_i = self.reconstruct_mul_data(data) # r_i / m_{i-1}
        data = self.layer(rm_i) # avg_pool( r_i / m_{i-1} )
        data = self.construct_mul_share(data) # avg_pool( r_i / m_{i-1} ) .* m_{i}
        self.send_he(data)
        self.stat.time_offline += time.time() - t
        return rm_i
        
    def online(self) -> torch.Tensor:
        t = time.time()
        data = self.recv_plain() # xmr_i = x_i * m_{i-1} - r_i
        xrm_i = self.reconstruct_mul_data(data) # x_i - r_i / m_{i-1}
        data = self.layer(xrm_i) # avg_pool( x_i - r_i / m_{i-1} )
        data = self.construct_mul_share(data) # avg_pool( x_i - r_i / m_{i-1} ) .* m_{i}
        self.send_plain(data)
        self.stat.time_online += time.time() - t
        return xrm_i
    