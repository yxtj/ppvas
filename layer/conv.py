from layer.base import LayerClient, LayerServer

from socket import socket
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
        # TODO: construct HE layer
    
    def setup(self, mlast:torch.Tensor, m_other:torch.Tensor=None, identity_m:bool=False) -> None:
        assert m_other is None
        super().setup(mlast, m_other=m_other, identity_m=identity_m)
        t = time.time()
        if not identity_m:
            self.set_m_positive()
        self.stat.time_offline += time.time() - t
    
    def offline(self) -> torch.Tensor:
        t = time.time()
        r_i = self.recv_he()
        data = self.reconstruct_mul_data(r_i) # r_i / m_{i-1}
        data = self.run_layer_offline(data) # W_i * r_i / m_{i-1}
        data = self.construct_mul_share(data) # W_i * r_i / m_{i-1} .* m_{i}
        self.send_he(data)
        self.stat.time_offline += time.time() - t
        return r_i
        
    def online(self) -> torch.Tensor:
        t = time.time()
        xmr_i = self.recv_plain() # xmr_i = x_i * m_{i-1} - r_i
        data = self.reconstruct_mul_data(xmr_i) # x_i - r_i / m_{i-1}
        data = self.layer(data) # W_i * (x_i - r_i / m_{i-1})
        data = self.construct_mul_share(data) # W_i * (x_i - r_i / m_{i-1}) .* m_{i}
        self.send_plain(data)
        self.stat.time_online += time.time() - t
        return xmr_i

