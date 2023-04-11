from .ptobase import ProBaseServer, ProBaseClient

import torch
import numpy as np
import comm
from .sshare import gen_add_share, gen_mul_share

__ALL__ = ['ProtocolClient', 'ProtocolServer']

class ProtocolClient(ProBaseClient):
    def setup(self, sshape: tuple, rshape: tuple, **kwargs) -> None:
        super().setup(sshape, rshape, **kwargs)
        self.r = gen_add_share(sshape)
    
    def send_offline(self, data: torch.Tensor) -> None:
        self.stat.byte_offline_send += self._basic_he_send_(data)

    def recv_offline(self) -> torch.Tensor:
        data, nbytes = self._basic_he_recv_()
        self.pre = data
        self.stat.byte_offline_recv += nbytes
        return data

    def send_online(self, data: torch.Tensor) -> None:
        data = data + self.r
        self.stat.byte_online_send += comm.send_torch(self.socket, data)

    def recv_online(self) -> torch.Tensor:
        data, nbyte = comm.recv_torch(self.socket)
        self.stat.byte_online_recv += nbyte
        data = data + self.pre
        return data


class ProtocolServer(ProBaseServer):
    def setup(self, sshape: tuple, rshape: tuple, mlast: torch.Tensor, **kwargs) -> None:
        super().setup(sshape, rshape, **kwargs)
        self.mlast = mlast
        self.m = gen_mul_share(sshape)
    
    def recv_offline(self) -> np.ndarray:
        data, nbytes = self._basic_he_recv_()
        self.pre = data
        self.stat.byte_offline_recv += nbytes
        return data
        
    def send_offline(self, data: np.ndarray) -> None:
        self.stat.byte_offline_send += self._basic_he_send_(data)

    def recv_online(self) -> torch.Tensor:
        data, nbyte = comm.recv_torch(self.socket)
        self.stat.byte_online_recv += nbyte
        data = data/self.mlast
        return data
    
    def send_online(self, data: torch.Tensor) -> None:
        data = data*self.m
        self.stat.byte_online_send += comm.send_torch(self.socket, data)
