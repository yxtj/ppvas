import socket
import torch
import numpy as np
from Pyfhel import Pyfhel
import struct
from typing import Union

import comm
from setting import USE_HE
from layer_basic.stat import Stat


class ProtocolBase():
    def __init__(self, s: socket.socket, stat: Stat, he: Pyfhel):
        self.socket = s
        self.stat = stat
        self.he = he
        self.sshape = None
        self.rshape = None
        
    def setup(self, sshape:tuple, rshape:tuple, **kwargs) -> None:
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

    def _basic_he_send_(self, data:Union[torch.Tensor, np.ndarray]) -> int:
        if USE_HE:
            # actual send with HE
            # encrypt and turn to numpy
            nbytes = comm.send_he_matrix(self.socket, data, self.he)
        else:
            # simulate with plain
            nbytes = comm.send_torch(self.socket, data)
        return nbytes
        
    def _basic_he_recv_(self) -> tuple[Union[torch.Tensor, np.ndarray], int]:
        if USE_HE:
            # actual send with HE
            data, nbytes = comm.recv_he_matrix(self.socket, self.he)
            # decrypt
            # turn to torch
        else:
            # simulate with plain
            data, nbytes = comm.recv_torch(self.socket)
        return data, nbytes


# Client workflow: send -> recv
class ProBaseClient(ProtocolBase):
    def __init__(self, s: socket.socket, stat: Stat, he: Pyfhel):
        super().__init__(s, stat, he)
        self.r = None
        self.pre = None
    
    def setup(self, sshape:tuple, rshape:tuple, **kwargs) -> None:
        super().setup(sshape, rshape)
        if USE_HE:
            b_ctx = self.he.to_bytes_context()
            self.socket.sendall(struct.pack('!i', len(b_ctx)) + b_ctx)


# Server workflow: recv -> process -> send
class ProBaseServer(ProtocolBase):
    def __init__(self, s: socket.socket, stat: Stat, he: Pyfhel):
        super().__init__(s, stat, he)
        self.mlast = None
        self.m = None
    
    def setup(self, sshape: tuple, rshape: tuple, **kwargs) -> None:
        super().setup(sshape, rshape)
        if USE_HE:
            len = struct.unpack('!i', self.socket.recv(4))[0]
            b_ctx = self.socket.recv(len)
            self.he.from_bytes_context(b_ctx)
    

