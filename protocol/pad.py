from .ptobase import ProBaseServer, ProBaseClient

import torch
import numpy as np
import comm
from setting import USE_HE


class ProtocolClient(ProBaseClient):
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


class ProtocolServer(ProBaseServer):
    def recv_offline(self) -> torch.Tensor:
        if USE_HE:
            data, nbytes = comm.recv_he_matrix(self.socket, self.he)
        else:
            data, nbytes = comm.recv_torch(self.socket)
        self.stat.byte_offline_recv += nbytes
        return data
        
    def send_offline(self, data: np.ndarray) -> None:
        if USE_HE:
            self.stat.byte_offline_send += comm.send_he_matrix(self.socket, data, self.he)
        else:
            self.stat.byte_offline_send += comm.send_torch(self.socket, data)

    def recv_online(self) -> torch.Tensor:
        data, nbyte = comm.recv_torch(self.socket)
        self.stat.byte_online_recv += nbyte
        data = data/self.mlast
        return data
    
    def send_online(self, data: torch.Tensor) -> None:
        data = data*self.m
        self.stat.byte_online_send += comm.send_torch(self.socket, data)
