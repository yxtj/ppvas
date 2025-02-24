from .base import LayerClient, LayerServer

from socket import socket
from typing import Union
import numpy as np
import time
import torch
from Pyfhel import Pyfhel


class MaxPoolClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel,
                 layer:torch.nn.MaxPool2d) -> None:
        super().__init__(socket, ishape, oshape, he)
        self.layer = layer
        
    def online(self, xm) -> torch.Tensor:
        t = time.time()
        if self.is_input_layer:
            xm = (xm, xm)
        self.send_online(xm[0], xm[1])
        data = self.recv_online() # x .* mp
        ya = self.layer(data[0]) # max_pool(x) .* m
        yb = self.layer(data[1])
        if self.is_output_layer:
            data = ya
        else:
            data = (ya, yb)
        self.stat.time_online += time.time() - t
        return data


class MaxPoolServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, torch.nn.MaxPool2d)
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
    
    def setup(self, last_lyr: LayerServer, m: Union[torch.Tensor, float, int]=None, **kwargs) -> None:
        super().setup(last_lyr, m)
        t = time.time()
        # update ma and mb (the mp of the smp protocol)
        block = torch.ones(self.stride_shape)
        self.ma = torch.kron(self.ma, block) # kronecker product
        self.mb = torch.kron(self.mb, block) # kronecker product
        self.stat.time_offline += time.time() - t
        
    def cut_input(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        h = self.oshape[-2] * self.stride_shape[0]
        w = self.oshape[-1] * self.stride_shape[1]
        if (h, w) == x.shape[-2:]:
            return x
        else:
            return x[..., :h, :w]
    
    def offline(self) -> np.ndarray:
        t = time.time()
        rm = self.recv_offline() # rm = r_i / m_{i-1}
        data = self.cut_input(rm)
        self.send_offline(data)
        self.stat.time_offline += time.time() - t
        return rm
        
    def online(self) -> torch.Tensor:
        t = time.time()
        xrm = self.recv_online() # xrm = (x_i - r_i / m_{i-1})
        data = self.cut_input(xrm)
        self.send_online(data)
        self.stat.time_online += time.time() - t
        return xrm
    