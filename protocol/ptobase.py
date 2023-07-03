import socket
import torch
import numpy as np
from Pyfhel import Pyfhel
import struct
from typing import Union
import heutil

import comm
from layer_basic.stat import Stat
from .sshare import gen_add_share, gen_mul_share
from setting import USE_HE

NumberType = Union[int, float, torch.Tensor, np.ndarray]

class ProtocolBase():
    def __init__(self, s: socket.socket, stat: Stat, he: Pyfhel):
        self.socket = s
        self.stat = stat
        self.he = he
        self.ishape = None
        self.oshape = None
        
    def setup(self, ishape:tuple, oshape:tuple, **kwargs) -> None:
        self.ishape = ishape
        self.oshape = oshape

    def send_offline(self, data: Union[torch.Tensor, np.ndarray]) -> None:
        raise NotImplementedError
    
    def recv_offline(self) -> Union[torch.Tensor, np.ndarray]:
        raise NotImplementedError
    
    def send_online(self, data: torch.Tensor) -> None:
        raise NotImplementedError
    
    def recv_online(self) -> torch.Tensor:
        raise NotImplementedError

    def _gen_add_share_(self, v: NumberType, shape: tuple) -> torch.Tensor:
        if v is not None:
            assert isinstance(v, (int, float, torch.Tensor, np.ndarray))
            if isinstance(v, torch.Tensor):
                assert v.shape == shape
                r = v
            elif isinstance(v, np.ndarray):
                assert v.shape == shape
                r = torch.from_numpy(v)
            elif isinstance(v, (int, float)):
                r = torch.zeros(shape).fill_(v)
        else:
            r = gen_add_share(shape)
        return r
    
    def _gen_mul_share_(self, v: NumberType, shape: tuple) -> torch.Tensor:
        if v is not None:
            assert isinstance(v, (int, float, torch.Tensor, np.ndarray))
            if isinstance(v, torch.Tensor):
                assert v.shape == shape
                r = v
            elif isinstance(v, np.ndarray):
                assert v.shape == shape
                r = torch.from_numpy(v)
            elif isinstance(v, (int, float)):
                if r == 0:
                    r = torch.zeros(shape)
                elif v == 1:
                    r = torch.ones(shape)
                else:
                    r= torch.zeros(shape).fill_(v)
        else:
            r = gen_mul_share(shape)
        return r


# Client workflow: send -> recv
class ProBaseClient(ProtocolBase):
    def __init__(self, s: socket.socket, stat: Stat, he: Pyfhel):
        super().__init__(s, stat, he)
        self.r = None
        self.pre = None
    
    def setup(self, ishape:tuple, oshape:tuple, r: NumberType=None, **kwargs) -> None:
        super().setup(ishape, oshape)
        self.r = self._gen_add_share_(r, ishape)
        if USE_HE:
            b_ctx = self.he.to_bytes_context()
            self.socket.sendall(struct.pack('!i', len(b_ctx)) + b_ctx)
            self.stat.byte_offline_send += 4 + b_ctx

    def send_offline(self, data: torch.Tensor) -> None:
        if USE_HE:
            data = heutil.encrypt(self.he, data)
            nbyte = comm.send_he_matrix(self.socket, data, self.he)
        else:
            nbyte = comm.send_torch(self.socket, data)
        self.stat.byte_offline_send += nbyte
    
    def recv_offline(self) -> torch.Tensor:
        if USE_HE:
            data, nbyte = comm.recv_he_matrix(self.socket, self.he)
            data = heutil.decrypt(self.he, data)
        else:
            data, nbyte = comm.recv_torch(self.socket)
        self.pre = data
        self.stat.byte_offline_recv += nbyte
        return data


# Server workflow: recv -> process -> send
class ProBaseServer(ProtocolBase):
    def __init__(self, s: socket.socket, stat: Stat, he: Pyfhel):
        super().__init__(s, stat, he)
        self.mlast = None
        self.m = None
    
    def setup(self, ishape: tuple, oshape: tuple, s: NumberType, 
              mlast: NumberType=1, m: NumberType=None, **kwargs) -> None:
        '''
        if m is None, then m is randomly generated (>0)
        '''
        super().setup(ishape, oshape)
        self.s = self._gen_add_share_(s, oshape)
        self.mlast = mlast
        self.m = self._gen_mul_share_(m)
        if USE_HE:
            len = struct.unpack('!i', self.socket.recv(4))[0]
            b_ctx = self.socket.recv(len)
            self.he.from_bytes_context(b_ctx)
            self.stat.byte_offline_recv += 4 + len
    
    def recv_offline(self) -> Union[torch.Tensor, np.ndarray]:
        if USE_HE:
            data, nbyte = comm.recv_he_matrix(self.socket, self.he)
        else:
            data, nbyte = comm.recv_torch(self.socket)
        self.stat.byte_offline_recv += nbyte
        return data
        
    def send_offline(self, data: Union[torch.Tensor, np.ndarray]) -> None:
        if USE_HE:
            nbyte = comm.send_he_matrix(self.socket, data, self.he)
        else:
            nbyte = comm.send_torch(self.socket, data)
        self.stat.byte_offline_send += nbyte
    