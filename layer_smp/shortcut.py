from .base import LayerClient, LayerServer

from socket import socket
from typing import Union
import time
import torch
# import torch.nn as nn
from torch_extension.shortcut import ShortCut
from Pyfhel import Pyfhel

class ShortCutClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        assert ishape == oshape
        super().__init__(socket, ishape, oshape, he)
    
    def setup(self, r_other: torch.Tensor, **kwargs) -> None:
        assert self.ishape == r_other.shape
    
class ShortCutServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, ShortCut)
        assert ishape == oshape
        super().__init__(socket, ishape, oshape, layer)
        self.other_offset = layer.otherlayer
        # self.mj = None
    
    def setup(self, last_lyr: LayerServer, m: Union[torch.Tensor, float, int]=None, **kwargs) -> None:
        # assert 'm_other' in kwargs, 'm_other is not provided'
        t = time.time()
        super().setup(last_lyr, m)
        # m_other = kwargs['m_other']
        # assert isinstance(m_other, torch.Tensor)
        # assert self.ishape == m_other.shape
        # self.mj = m_other
        self.stat.time_offline += time.time() - t
        # print("shortcut setup: mi={}, mj={}, m={}".format(self.mlast, self.mj, self.m))
    
    def offline(self, rmj) -> torch.Tensor:
        t = time.time()
        data = self.recv_he() # r_i
        # print("r_i={}, r_j={}".format(ri, rj))
        rmi = self.reconstruct_mul_data(data) # r_i / m_{i-1}
        data = rmi + rmj
        data = self.construct_mul_share(data) # (r_i / m_{i-1} + r_j / m_{j-1}) * m_i
        self.send_he(data)
        self.stat.time_offline += time.time() - t
        return rmi
        
    def online(self, xrm_j) -> torch.Tensor:
        t = time.time()
        data = self.recv_plain() # xmr_i = x_i * m_{i-1} - r_i
        xrm_i = self.reconstruct_mul_data(data) # x_i - r_i / m_{i-1}
        data = xrm_i + xrm_j
        data = self.construct_mul_share(data) # (x_i - r_i / m_{i-1} + x_j - r_j / m_{j-1}) * m_i
        self.send_plain(data)
        self.stat.time_online += time.time() - t
        return xrm_i
    