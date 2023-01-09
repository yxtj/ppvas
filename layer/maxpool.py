from layer.base import LayerClient, LayerServer
from comm.util import send_torch, recv_torch

from socket import socket
import torch
import torch.nn as nn

class MaxPoolClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, nn.MaxPool2d)
        super().__init__(socket, ishape, oshape)
        self.layer = layer
        
    def online(self, xm) -> torch.Tensor:
        send_torch(self.socket, xm)
        data = recv_torch(self.socket)
        data = self.reconstruct_add_data(data)
        data = self.layer(data)
        return data

class MaxPoolServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape:
        tuple, layer: torch.nn.Module, m_last: torch.Tensor) -> None:
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
        assert ishape[-2]*self.stride_shape[0] == oshape[-2] and ishape[-1]*self.stride_shape[1] == oshape[-1]
        super().__init__(socket, ishape, oshape, layer, m_last)
        # set m
        self.set_m_positive()
        # set mp
        if isinstance(layer.stride, 1):
            stride_shape = (layer.stride, layer.stride)
        else:
            stride_shape = layer.stride
        block = torch.ones(stride_shape)
        self.mp = torch.kron(self.m, block) # kronecker product
        
    def offline(self) -> None:
        data = recv_torch(self.socket)
        data = self.layer(data)
        data = self.construct_mul_share(data, self.mp)
        send_torch(self.socket, data)
        
    def online(self) -> None:
        data = recv_torch(self.socket)
        data = self.reconstruct_mul_data(data)
        data = self.layer(data)
        data = self.construct_mul_share(data, self.mp)
        send_torch(self.socket, data)
    