from dataclasses import dataclass
from socket import socket
import time
import torch
import numpy as np
from Pyfhel import Pyfhel

import comm.util

__ADD_SHARE_RANGE__ = 10
__MUL_SHARE_RANGE__ = 10
__POSITIVE_EPS__ = 0.01

@dataclass
class Stat:
    time_offline: float
    byte_offline_send: float
    byte_offline_recv: float
    # time_offline_send: float
    # time_offline_recv: float
    time_online: float
    byte_online_send: float
    byte_online_recv: float
    # time_online_send: float
    # time_online_recv: float
    
    def __add__(self, other):
        return Stat(
            self.time_offline + other.time_offline,
            self.byte_offline_send + other.byte_offline_send,
            self.byte_offline_recv + other.byte_offline_recv,
            self.time_online + other.time_online,
            self.byte_online_send + other.byte_online_send,
            self.byte_online_recv + other.byte_online_recv,
        )


class LayerCommon():
    def __init__(self, socket:socket, ishape:tuple, oshape:tuple, he:Pyfhel) -> None:
        self.socket = socket
        self.ishape = ishape
        self.oshape = oshape
        self.he = he
        self.stat = Stat(0, 0, 0, 0, 0, 0)
    
    def send_plain(self, data:torch.Tensor) -> None:
        self.stat.byte_online_send += comm.util.send_torch(self.socket, data)
    
    def recv_plain(self) -> torch.Tensor:
        data, nbyte = comm.util.recv_torch(self.socket)
        self.stat.byte_online_recv += nbyte
        return data

    def send_he(self, data:np.ndarray) -> None:
        # simulate with plain
        self.stat.byte_offline_send += comm.util.send_torch(self.socket, data)
        # actual send with HE
        # self.stat.byte_offline_send += comm.util.send_he_matrix(self.socket, data, self.he)
    
    def recv_he(self) -> np.ndarray:
        # simulate with plain
        data, nbyte = comm.util.recv_torch(self.socket)
        self.stat.byte_offline_recv += nbyte
        return data
        # actual send with HE
        data, nbytes = comm.util.recv_he_matrix(self.socket, self.he)
        self.stat.byte_offline_recv += nbytes
        return data

class LayerClient(LayerCommon):
    def __init__(self, socket:socket, ishape:tuple, oshape:tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)
        self.r = None
        self.pre = None
    
    def setup(self, **kwargs):
        return
    
    def offline(self) -> None:
        t = time.time()
        self.set_r()
        # print("r", self.r)
        self.send_he(self.r)
        data = self.recv_he()
        self.pre = data
        # print("pre", self.pre)
        self.stat.time_offline += time.time() - t
    
    def online(self, xm) -> torch.Tensor:
        t = time.time()
        # print("xm", xm)
        data = self.construct_add_share(xm)
        # print("xm-r", data)
        self.send_plain(data)
        data = self.recv_plain()
        # print("w(x-r/m)m", data)
        data = self.reconstruct_add_data(data)
        # print("wxm", data)
        self.stat.time_online += time.time() - t
        return data
    
    def set_r(self):
        self.r = __ADD_SHARE_RANGE__*torch.rand(self.ishape) - __ADD_SHARE_RANGE__/2 # [-5, 5)
        # self.r = torch.zeros(self.ishape)
        # self.r = torch.zeros(self.ishape) + np.random.randint(1, 6)*0.1
    
    def construct_add_share(self, data):
        return data - self.r
    
    def reconstruct_add_data(self, share):
        return share + self.pre
    
    
class LayerServer(LayerCommon):
    def __init__(self, socket:socket, ishape:tuple, oshape:tuple, layer:torch.nn.Module) -> None:
        super().__init__(socket, ishape, oshape, Pyfhel())
        self.layer = layer
        self.mlast = None
        self.m = None
    
    def setup(self, mlast:torch.Tensor) -> None:
        if isinstance(mlast, (int, float)):
            mlast = torch.zeros(self.ishape) + mlast
        else:
            assert mlast.shape == self.ishape
        self.mlast = mlast
        # set m in child class
    
    def offline(self) -> torch.Tensor:
        raise NotImplementedError
    
    def online(self) -> torch.Tensor:
        raise NotImplementedError

    def run_layer_offline(self, data:torch.Tensor) -> torch.Tensor:
        bias = self.layer.bias
        self.layer.bias = None
        data = self.layer(data)
        self.layer.bias = bias
        return data

    def set_m_any(self) -> None:
        t = __MUL_SHARE_RANGE__*torch.rand(self.oshape)
        f = t < __POSITIVE_EPS__
        t[f] += __POSITIVE_EPS__
        self.m = t # [-5, -eps) U (eps, 5)
        # self.m = torch.ones(self.oshape)
    
    def set_m_positive(self) -> None:
        t = __MUL_SHARE_RANGE__*torch.rand(self.oshape) + __POSITIVE_EPS__ # avoid zero
        self.m = t # [eps, 10+eps)
        # self.m = torch.ones(self.oshape)
    
    def construct_mul_share(self, data, m = None):
        '''
        Construct multiplicative share for sending online data.
        Multiply m to data.
        '''
        if m is None:
            m = self.m
        return data*m
    
    def reconstruct_mul_data(self, share, mlast = None):
        '''
        Reconstruct data from the multiplicative share for received online data.
        Divide mlast from share.
        '''
        if mlast is None:
            mlast = self.mlast
        return share / mlast


# local layer specialization

class LocalLayerClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)
    
    def offline(self) -> None:
        return

    def online(self, xm) -> torch.Tensor:
        raise NotImplementedError

class LocalLayerServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        super().__init__(socket, ishape, oshape, layer)
    
    def setup(self, mlast: torch.Tensor) -> None:
        super().setup(mlast)
        self.m = mlast
    
    def offline(self) -> None:
        return
    
    def online(self) -> None:
        return

