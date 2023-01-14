from layer.base import LayerClient, LayerServer

from socket import socket
import time
import torch
# import torch.nn as nn
from torch_extension.shortcut import ShortCut
from Pyfhel import Pyfhel

class ShortCutClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        assert ishape == oshape
        super().__init__(socket, ishape, oshape, he)
        self.rj = None
    
    def setup(self, r_other: torch.Tensor) -> None:
        assert self.ishape == r_other.shape
        self.rj = r_other
    
class ShortCutServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, ShortCut)
        assert ishape == oshape
        super().__init__(socket, ishape, oshape, layer)
        self.other_offset = layer.otherlayer
        self.mj = None
    
    def setup(self, m_last: torch.Tensor, m_other: torch.Tensor) -> None:
        super().setup(m_last)
        assert self.ishape == m_other.shape
        t = time.time()
        self.set_m_any()
        self.mj = m_other
        self.stat.time_offline += time.time() - t
    
    def offline(self, rj) -> torch.Tensor:
        t = time.time()
        ri = self.recv_he()
        ci = self.reconstruct_mul_data(ri)
        cj = self.reconstruct_mul_data(rj, self.mj)
        data = ci + cj
        data = self.construct_mul_share(data)
        self.send_he(data)
        self.stat.time_offline += time.time() - t
        return ri
        
    def online(self, xmr_j) -> torch.Tensor:
        t = time.time()
        cj = self.reconstruct_mul_data(xmr_j, self.mj)
        xmr_i = self.recv_plain()
        ci = self.reconstruct_mul_data(xmr_i)
        data = ci + cj
        data = self.construct_mul_share(data)
        self.send_plain(data)
        self.stat.time_online += time.time() - t
        return xmr_i
    