from .ptobase import ProBaseServer, ProBaseClient

import torch
import numpy as np
import comm
from .sshare import gen_add_share, gen_mul_share

__ALL__ = ['ProtocolClient', 'ProtocolServer']

class ProtocolClient(ProBaseClient):
    def setup(self, ishape: tuple, oshape: tuple, **kwargs) -> None:
        super().setup(ishape, oshape, **kwargs)
        self.r = gen_add_share(ishape)
    
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
    def setup(self, ishape: tuple, oshape: tuple, mlast: torch.Tensor, **kwargs) -> None:
        '''
        If 'm' is given, use it. Otherwise, generate a random m.
        '''
        super().setup(ishape, oshape, mlast, **kwargs)
        if 'm' in kwargs and kwargs['m'] is not None:
            m = kwargs['m']
            if isinstance(m, torch.Tensor):
                assert m.shape == ishape
                self.m = m
            elif isinstance(m, np.ndarray):
                assert m.shape == ishape
                self.m = torch.from_numpy(m)
            elif isinstance(m, (int, float)):
                if m == 0:
                    self.m = torch.zeros(ishape)
                elif m == 1:
                    self.m = torch.ones(ishape)
                else:
                    self.m = torch.zeros(ishape).fill_(m)
            else:
                raise TypeError("m should be torch.Tensor or np.ndarray")
        else:
            self.m = gen_mul_share(ishape)
    
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
