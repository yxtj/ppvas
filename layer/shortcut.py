from layer.base import LayerClient, LayerServer
from comm.util import send_torch, recv_torch

from socket import socket
import torch
import torch.nn as nn
from torch_extension.shortcut import ShortCut

class ShortCutClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple,
                 rj: torch.Tensor) -> None:
        assert ishape == oshape
        assert ishape == rj.shape
        super().__init__(socket, ishape, oshape)
        self.rj = rj
        
class ShortCutServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple,
                 layer: torch.nn.Module, m_last: torch.Tensor,
                 m_other:torch.Tensor) -> None:
        assert isinstance(layer, ShortCut)
        assert ishape == oshape
        assert ishape == m_other.shape
        super().__init__(socket, ishape, oshape, layer, m_last)
        self.set_m_any()
        self.mj = m_other
        
    def offline(self, rj) -> None:
        ri = recv_torch(self.socket)
        ci = self.reconstruct_mul_data(ri)
        cj = self.reconstruct_mul_data(rj, self.mj)
        data = ci + cj
        data = self.construct_mul_share(data)
        send_torch(self.socket, data)
        
    def online(self, xmr_j) -> None:
        cj = self.reconstruct_mul_data(xmr_j, self.mj)
        xmr_i = recv_torch(self.socket)
        ci = self.reconstruct_mul_data(xmr_i)
        data = ci + cj
        data = self.construct_mul_share(data)
        send_torch(self.socket, data)
    