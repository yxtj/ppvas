from layer.base import LayerClient, LayerServer
from comm.util import send_torch, recv_torch

from socket import socket
import torch
# import torch.nn as nn
from torch_extension.shortcut import ShortCut

class ShortCutClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple) -> None:
        assert ishape == oshape
        super().__init__(socket, ishape, oshape)
        self.rj = None
    
    def setup(self, offset:int, r_other: torch.Tensor) -> None:
        assert self.ishape == r_other.shape
        self.other_offset = offset
        self.rj = r_other
    
class ShortCutServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple,
                 layer: torch.nn.Module, m_last: torch.Tensor) -> None:
        assert isinstance(layer, ShortCut)
        assert ishape == oshape
        super().__init__(socket, ishape, oshape, layer, m_last)
        self.other_offset = layer.otherlayer
        self.set_m_any()
        self.mj = None
    
    def setup(self, m_other):
        assert self.ishape == m_other.shape
        self.mj = m_other
    
    def offline(self, rj) -> torch.Tensor:
        ri = recv_torch(self.socket)
        ci = self.reconstruct_mul_data(ri)
        cj = self.reconstruct_mul_data(rj, self.mj)
        data = ci + cj
        data = self.construct_mul_share(data)
        send_torch(self.socket, data)
        return ri
        
    def online(self, xmr_j) -> torch.Tensor:
        cj = self.reconstruct_mul_data(xmr_j, self.mj)
        xmr_i = recv_torch(self.socket)
        ci = self.reconstruct_mul_data(xmr_i)
        data = ci + cj
        data = self.construct_mul_share(data)
        send_torch(self.socket, data)
        return xmr_i
    