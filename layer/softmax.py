from .base import LocalLayerClient, LocalLayerServer

from socket import socket
from typing import Union
# import numpy as np
import time
import torch
from Pyfhel import Pyfhel

class SoftmaxClient(LocalLayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)
        self.layer = torch.nn.Softmax(1)
    
    def online(self, xm) -> torch.Tensor:
        t = time.time()
        assert self.is_output_layer
        data = self.layer(xm[0])
        self.stat.time_online += time.time() - t
        return data
    

class SoftmaxServer(LocalLayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, torch.nn.Softmax)
        super().__init__(socket, ishape, oshape, layer)
    
    def setup(self, last_lyr: LocalLayerServer, m: Union[torch.Tensor, float, int]=None, **kwargs) -> None:
        if last_lyr is None:
            super().setup(last_lyr, m)
        else:
            # since softmax is the last layer in the neural network, we just copy the last layer's m, h, ma, mb
            # the protocol guarantees that the previous layer's m is 1
            m = last_lyr.m
            h = last_lyr.h
            ma = last_lyr.ma
            mb = last_lyr.mb
            super().setup(last_lyr, m, h=h, ma=ma, mb=mb)

