from .base import LayerClient, LayerServer

from socket import socket
from typing import Union
import time
import torch
import torch.nn as nn
from Pyfhel import Pyfhel

class MaxPoolClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel,
                 layer:torch.nn.MaxPool2d) -> None:
        super().__init__(socket, ishape, oshape, he)
        self.layer = layer
        
    def online(self, xm) -> torch.Tensor:
        t = time.time()
        # print("xm", xm)
        data = self.construct_add_share(xm)
        # print("xm-r", data)
        self.send_plain(data)
        data = self.recv_plain()
        # print("(x-r/m) * mp", data)
        data = self.reconstruct_add_data(data)
        # print("x * mp", data)
        data = self.layer(data)
        # print("x * m", data)
        self.stat.time_online += time.time() - t
        return data

class MaxPoolServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, nn.MaxPool2d)
        # kernel_size must be no greater than stride
        if isinstance(layer.kernel_size, int):
            assert layer.kernel_size <= layer.stride
        elif isinstance(layer.kernel_size, tuple):
            assert len(layer.kernel_size) == 2 and len(layer.stride) == 2
            assert layer.kernel_size[0] <= layer.stride[0] and layer.kernel_size[1] <= layer.stride[1]
        else:
            raise ValueError("kernel_size must be int or tuple")
        assert layer.padding == 0
        assert layer.dilation == 1
        
        if isinstance(layer.stride, int):
            stride_shape = (layer.stride, layer.stride)
        else:
            stride_shape = layer.stride
        assert ishape[-2]//stride_shape[0] == oshape[-2] and ishape[-1]//stride_shape[1] == oshape[-1]
        
        super().__init__(socket, ishape, oshape, layer)
        self.stride_shape = stride_shape
        self.mp = None
    
    def setup(self, last_lyr: LayerServer, m: Union[torch.Tensor, float, int]=None, **kwargs) -> None:
        t = time.time()
        super().setup(last_lyr, m)
        # set mp
        block = torch.ones(self.stride_shape)
        self.mp = torch.kron(self.m, block) # kronecker product
        # print("mp", self.mp)
        self.stat.time_offline += time.time() - t
        
    def cut_input(self, x: torch.Tensor) -> torch.Tensor:
        h = self.oshape[-2] * self.stride_shape[0]
        w = self.oshape[-1] * self.stride_shape[1]
        if (h, w) == x.shape[-2:]:
            return x
        else:
            return x[..., :h, :w]
    
    def offline(self) -> torch.Tensor:
        t = time.time()
        data = self.recv_he() # r_i
        # print("r_i", r_i)
        rm = self.reconstruct_mul_data(data) # r_i / m_{i-1}
        # print("r/m", data)
        data = self.cut_input(rm)
        data = self.construct_mul_share(data, self.mp) # r_i / m_{i-1} .* m^p_{i}
        # print("r/m * mp", data)
        self.send_he(data)
        self.stat.time_offline += time.time() - t
        return rm
        
    def online(self) -> torch.Tensor:
        t = time.time()
        data = self.recv_plain() # xmr_i = x_i * m_{i-1} - r_i
        # print("xmr_i", xmr_i)
        xrm = self.reconstruct_mul_data(data) # x_i - r_i / m_{i-1}
        # print("x-r/m", data)
        data = self.cut_input(xrm)
        data = self.construct_mul_share(data, self.mp) # (x_i - r_i / m_{i-1}) .* m^p_{i}
        # print("(x-r/m) * mp", data)
        self.send_plain(data)
        self.stat.time_online += time.time() - t
        return xrm
    