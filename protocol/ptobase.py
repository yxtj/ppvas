import socket
import torch
import numpy as np
from Pyfhel import Pyfhel
import struct
from typing import Union

from setting import USE_HE
from layer_basic.stat import Stat
from .sshare import gen_add_share, gen_mul_share


class ProtocolBase():
    def __init__(self, s: socket.socket, stat: Stat, he: Pyfhel):
        self.socket = s
        self.stat = stat
        self.he = he
        self.sshape = None
        self.rshape = None
        
    def setup(self, sshape:tuple, rshape:tuple) -> None:
        self.sshape = sshape
        self.rshape = rshape

    def send_offline(self, data: Union[torch.Tensor, np.ndarray]) -> None:
        raise NotImplementedError
    
    def recv_offline(self) -> torch.Tensor:
        raise NotImplementedError
    
    def send_online(self, data: torch.Tensor) -> None:
        raise NotImplementedError
    
    def recv_online(self) -> torch.Tensor:
        raise NotImplementedError


# workflow: send -> recv
class ProClient(ProtocolBase):
    def __init__(self, s: socket.socket, stat: Stat, he: Pyfhel):
        super().__init__(s, stat, he)
        self.r = None
        self.pre = None
    
    def setup(self, sshape:tuple, rshape:tuple) -> None:
        super().setup(sshape, rshape)
        self.r = gen_add_share(sshape)
        if USE_HE:
            b_ctx = self.he.to_bytes_context()
            self.socket.sendall(struct.pack('!i', len(b_ctx)) + b_ctx)


# recv -> process -> send
class ProServer(ProtocolBase):
    def __init__(self, s: socket.socket, stat: Stat, he: Pyfhel):
        super().__init__(s, stat, he)
        self.mlast = None
        self.m = None
    
    def setup(self, sshape: tuple, rshape: tuple, mlast: torch.Tensor) -> None:
        super().setup(sshape, rshape)
        self.mlast = mlast
        self.m = gen_mul_share(sshape)
        if USE_HE:
            len = struct.unpack('!i', self.socket.recv(4))[0]
            b_ctx = self.socket.recv(len)
            self.he.from_bytes_context(b_ctx)
    

