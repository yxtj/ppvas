from layer.base import LayerClient, LayerServer

from socket import socket
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
    
    def setup(self, mlast:torch.Tensor, *, m_other:torch.Tensor=None, identity_m:bool=False) -> None:
        assert m_other is None
        super().setup(mlast, m_other=m_other, identity_m=identity_m)
        t = time.time()
        if not identity_m:
            self.set_m_any()
        self.stat.time_offline += time.time() - t
        
    def offline(self) -> torch.Tensor:
        t = time.time()
        r_i = self.recv_he() # r_i
        data = self.reconstruct_mul_data(r_i) # r_i / m_{i-1}
        data = self.layer(data) # avg_pool( r_i / m_{i-1} )
        data = self.construct_mul_share(data) # avg_pool( r_i / m_{i-1} ) .* m_{i}
        self.send_he(data)
        self.stat.time_offline += time.time() - t
        return r_i
        
    def online(self) -> torch.Tensor:
        t = time.time()
        xmr_i = self.recv_plain() # xmr_i = x_i * m_{i-1} - r_i
        data = self.reconstruct_mul_data(xmr_i) # x_i - r_i / m_{i-1}
        data = self.layer(data) # avg_pool( x_i - r_i / m_{i-1} )
        data = self.construct_mul_share(data) # avg_pool( x_i - r_i / m_{i-1} ) .* m_{i}
        self.send_plain(data)
        self.stat.time_online += time.time() - t
        return xmr_i
    