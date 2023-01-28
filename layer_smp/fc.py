from .base import LayerClient, LayerServer

from socket import socket
from typing import Union
import time
import torch
import torch.nn as nn
from Pyfhel import Pyfhel

class FcClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)

class FcServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, nn.Linear)
        super().__init__(socket, ishape, oshape, layer)
    
    def setup(self, m_last: Union[torch.Tensor, float, int],
              m: Union[torch.Tensor, float, int]=None, **kwargs) -> None:
        t = time.time()
        print("fc: m_last", m_last.flatten()[0] if m_last is not None else None,
              "m:", m.flatten()[0] if m_last is isinstance(m, torch.Tensor) else m)
        super().setup(m_last, m)
        self.stat.time_offline += time.time() - t
    
    def offline(self) -> torch.Tensor:
        t = time.time()
        r_i = self.recv_he() # r_i
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
