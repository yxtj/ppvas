from socket import socket
from typing import Union
import numpy as np
import torch
from Pyfhel import Pyfhel

import comm
from .stat import Stat

from setting import USE_HE


class LayerCommon():
    def __init__(self, socket:socket, ishape:tuple, oshape:tuple, he:Pyfhel) -> None:
        self.socket = socket
        self.ishape = ishape
        self.oshape = oshape
        self.he = he
        self.stat = Stat()
    
    def send_plain(self, data:torch.Tensor) -> None:
        self.stat.byte_online_send += comm.send_torch(self.socket, data)
    
    def recv_plain(self) -> torch.Tensor:
        data, nbyte = comm.recv_torch(self.socket)
        self.stat.byte_online_recv += nbyte
        return data

    def send_he(self, data:Union[np.ndarray, torch.Tensor]) -> None:
        if USE_HE:
            # actual send with HE
            assert isinstance(data, np.ndarray)
            self.stat.byte_offline_send += comm.send_he_matrix(self.socket, data, self.he)
        else:
            # simulate with plain
            assert isinstance(data, torch.Tensor)
            self.stat.byte_offline_send += comm.send_torch(self.socket, data)
    
    def recv_he(self) -> np.ndarray:
        if USE_HE:
            # actual send with HE
            data, nbytes = comm.recv_he_matrix(self.socket, self.he)
        else:
            # simulate with plain
            data, nbytes = comm.recv_torch(self.socket)
        self.stat.byte_offline_recv += nbytes
        return data
