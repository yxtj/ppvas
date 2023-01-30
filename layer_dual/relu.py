from .base import LocalLayerClient, LocalLayerServer

from socket import socket
from typing import Union
# import numpy as np
import time
import torch
from Pyfhel import Pyfhel

class ReLUClient(LocalLayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)
        self.layer = torch.nn.ReLU()
    
    def online(self, xm) -> Union[torch.Tensor, torch.Tensor]:
        t = time.time()
        if self.is_input_layer:
            xm = (xm, xm)
        ya = self.layer(xm[0])
        yb = self.layer(xm[1])
        if self.is_output_layer:
            data = ya
        else:
            data = (ya, yb)
        self.stat.time_online += time.time() - t
        return data
    
class ReLUServer(LocalLayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, torch.nn.ReLU)
        super().__init__(socket, ishape, oshape, layer)
        
    def setup(self, last_lyr: LocalLayerServer, m: Union[torch.Tensor, float, int]=None, **kwargs) -> None:
        if last_lyr is None:
            super().setup(last_lyr, m)
        else:
            super().setup(last_lyr, last_lyr.m, h=last_lyr.h, ma=last_lyr.ma, mb=last_lyr.mb)
