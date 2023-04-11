from .base import LayerClient, LayerServer

from socket import socket
from typing import Union
import numpy as np
import time
import torch
from torch_extension.shortcut import ShortCut
from Pyfhel import Pyfhel

class ShortCutClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        assert ishape == oshape
        super().__init__(socket, ishape, oshape, he)
    
    def setup(self, r_other: torch.Tensor, **kwargs) -> None:
        super().setup()
        assert self.ishape == r_other.shape
    
class ShortCutServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, ShortCut)
        assert ishape == oshape
        super().__init__(socket, ishape, oshape, layer)
        self.other_offset = layer.otherlayer
    
    def offline(self, rm_j) -> torch.Tensor:
        t = time.time()
        rm_i = self.recv_offline() # r_i/m_{i-1}
        data = rm_i + rm_j # r_i / m_{i-1} + r_j / m_{j-1}
        self.send_offline(data)
        self.stat.time_offline += time.time() - t
        return rm_i
    
    def online(self, xmr_j) -> torch.Tensor:
        t = time.time()
        xrm_i = self.recv_online() # x_i - r_i / m_{i-1}
        data = xrm_i + xmr_j # (x_i + x_j) - (r_i / m_{i-1} - r_j / m_{j-1})
        self.send_online(data)
        self.stat.time_online += time.time() - t
        return xrm_i
    