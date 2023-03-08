from .ptobase import ProServer, ProClient

import torch
import numpy as np
import comm
from typing import Union

from setting import USE_HE


class ProtocolClient(ProClient):
    def setup(self, sshape: tuple, rshape: tuple) -> None:
        super().setup(sshape, rshape)
    
    def send_offline(self, data: torch.Tensor) -> None:
        if USE_HE:
            # encrypt
            # turn to numpy
            self.stat.byte_offline_send += comm.send_he_matrix(self.socket, data, self.he)
        else:
            self.stat.byte_offline_send += comm.send_torch(self.socket, data)

    def recv_offline(self) -> torch.Tensor:
        if USE_HE:
            data, nbytes = comm.recv_he_matrix(self.socket, self.he)
            # decrypt
            # turn to torch
        else:
            data, nbytes = comm.recv_torch(self.socket)
        self.stat.byte_offline_recv += nbytes
        self.pre = data
        return data

    def send_online(self, data: torch.Tensor) -> None:
        data = data + self.r
        self.stat.byte_online_send += comm.send_torch(self.socket, data)

    def recv_online(self) -> torch.Tensor:
        data, nbyte = comm.recv_torch(self.socket)
        self.stat.byte_online_recv += nbyte
        data  = data + self.pre
        return data


class ProtocolServer(ProServer):
    def __init__(self, s, stat, he):
        super().__init__(s, stat, he)
        self.offline_buffer = None # used to generate random data for offline phase
    
    def setup(self, sshape: tuple, rshape: tuple, mlast: torch.Tensor, Rlast: torch.Tensor) -> None:
        super().setup(sshape, rshape, mlast)
        assert len(Rlast) == np.prod(self.rshape)
        self.Rlast = Rlast
        n = np.prod(sshape)
        self.S = torch.randperm(n).reshape(self.sshape) # shuffle matrix
        self.R = torch.argsort(self.S.ravel()).reshape(self.sshape) # unshuffle matrix
    
    def confuse_data(self, data: torch.Tensor) -> torch.Tensor:
        '''
        Shuffle input data.
        Input: a tensor of shape "self.sshape".
        Output: a tensor of shape "self.sshape".
        '''
        res = data.ravel()[self.S].reshape(self.sshape)
        return res
    
    def clearify_data(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        '''
        Pick and unshuffle.
        Input: a tensor of shape "self.rshape".
        Output: a tensor of shape "self.rshape".
        '''
        #assert len(self.Rlast) == 2*np.prod(self.rshape) == np.prod(data.shape)
        res = data.ravel()[self.Rlast].reshape(self.rshape)
        return res
    
    def recv_offline(self) -> torch.Tensor:
        if USE_HE:
            data, nbytes = comm.recv_he_matrix(self.socket, self.he)
        else:
            data, nbytes = comm.recv_torch(self.socket)
        self.stat.byte_offline_recv += nbytes
        self.offline_buffer = data
        data = self.clearify_data(data)
        return data
        
    def send_offline(self, data: np.ndarray) -> None:
        data = self.confuse_data(data)
        if USE_HE:
            self.stat.byte_offline_send += comm.send_he_matrix(self.socket, data, self.he)
        else:
            self.stat.byte_offline_send += comm.send_torch(self.socket, data)

    def recv_online(self) -> torch.Tensor:
        data, nbyte = comm.recv_torch(self.socket)
        self.stat.byte_online_recv += nbyte
        data = self.clearify_data(data)
        data = data/self.mlast
        return data
    
    def send_online(self, data: torch.Tensor) -> None:
        data = data*self.m
        data = self.confuse_data(data)
        self.stat.byte_online_send += comm.send_torch(self.socket, data)
