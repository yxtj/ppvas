from socket import socket
import torch

from comm.util import send_torch, recv_torch

__ADD_SHARE_RANGE__ = 10
__MUL_SHARE_RANGE__ = 10
__POSITIVE_EPS__ = 0.01

class LayerClient():
    def __init__(self, socket:socket, ishape:tuple, oshape:tuple) -> None:
        self.socket = socket
        self.ishape = ishape
        self.oshape = oshape
        self.r = None
        self.pre = None
    
    def setup(self, **kwargs):
        return
    
    def offline(self) -> None:
        self.set_r()
        # TODO: change to HE (encrypt and send, receive and decrypt)
        send_torch(self.socket, self.r)
        data = recv_torch(self.socket)
        self.pre = data
    
    def online(self, xm) -> torch.Tensor:
        data = self.construct_add_share(xm)
        send_torch(self.socket, data)
        data = recv_torch(self.socket)
        data = self.reconstruct_add_data(data)
        return data
    
    def set_r(self):
        self.r = __ADD_SHARE_RANGE__*torch.rand(self.ishape) - __ADD_SHARE_RANGE__/2 # [-5, 5)
    
    def construct_add_share(self, data):
        return data - self.r
    
    def reconstruct_add_data(self, share):
        return share + self.pre
    
    
class LayerServer():
    def __init__(self, socket:socket, ishape:tuple, oshape:tuple,
                 layer:torch.nn.Module, mlast:torch.Tensor) -> None:
        self.socket = socket
        self.ishape = ishape
        self.oshape = oshape
        self.mlast = mlast
        assert mlast.shape == ishape
        self.layer = layer
        self.m = None
    
    def setup(self, **kwargs):
        return
    
    def offline(self) -> None:
        raise NotImplementedError
    
    def online(self) -> None:
        raise NotImplementedError

    def set_m_any(self):
        t = __MUL_SHARE_RANGE__*torch.rand(self.oshape)
        f = t < __POSITIVE_EPS__
        t[f] += __POSITIVE_EPS__
        self.m = t # [-5, -eps) U (eps, 5)
    
    def set_m_positive(self):
        t = __MUL_SHARE_RANGE__*torch.rand(self.oshape) + __POSITIVE_EPS__# avoid zero
        self.m = t # [eps, 10+eps)
    
    def construct_mul_share(self, data, m = None):
        '''
        Construct multiplicative share for sending online data.
        Multiply m to data.
        '''
        if m is None:
            m = self.m
        return data*m
    
    def reconstruct_mul_data(self, share, mlast = None):
        '''
        Reconstruct data from the multiplicative share for received online data.
        Divide mlast from share.
        '''
        if mlast is None:
            mlast = self.mlast
        return share / mlast
    