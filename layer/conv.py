from .base import LayerClient, LayerServer

from socket import socket
from typing import Union
import numpy as np
import time
import torch
from Pyfhel import Pyfhel

class ConvClient(LayerClient):
    def __init__(self, socket:socket, ishape:tuple, oshape:tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)

    
class ConvServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, torch.nn.Conv2d)
        super().__init__(socket, ishape, oshape, layer)
    
    def offline(self) -> np.ndarray:
        t = time.time()
        rm = self.recv_offline() # r'_i = r_i / m_{i-1}
        data = self.run_layer_offline(rm) # W_i * r'_i
        self.send_offline(data)
        self.stat.time_offline += time.time() - t
        return rm
    
    def online(self) -> torch.Tensor:
        t = time.time()
        xmr_i = self.recv_online() # xmr_i = x_i - r_i / m_{i-1}
        data = self.layer(xmr_i) # W_i * (x_i - r_i / m_{i-1})
        self.send_online(data)
        self.stat.time_online += time.time() - t
        return xmr_i

