from socket import socket
from typing import Union
import time
import torch
import numpy as np
from Pyfhel import Pyfhel

import comm
from layer_basic import LayerCommon, gen_add_share, gen_mul_share
from protocol import ProtocolClient, ProtocolServer

__all__ = ['LayerClient', 'LayerServer', 'LocalLayerClient', 'LocalLayerServer']


class LayerClient(LayerCommon):
    def __init__(self, socket:socket, ishape:tuple, oshape:tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)
        self.is_input_layer = False
        self.is_output_layer = False
        self.protocol = ProtocolClient(socket, self.stat, he)
    
    def setup(self, **kwargs):
        t = time.time()
        self.protocol.setup(self.ishape, self.oshape, **kwargs)
        self.stat.time_offline += time.time() - t
    
    def offline(self) -> None:
        t = time.time()
        # print("r", self.protocol.r)
        self.protocol.send_offline(self.r)
        self.protocol.recv_offline()
        # print("pre", self.protocol.pre)
        self.stat.time_offline += time.time() - t
    
    def online(self, xmdual) -> tuple[torch.Tensor, torch.Tensor]:
        t = time.time()
        # print("xm", xm)
        # if isinstance(xmdual, torch.Tensor):
        if self.is_input_layer:
            xmdual = (xmdual, xmdual)
        self.protocol.send_online(xmdual[0], xmdual[1])
        data = self.protocol.recv_online()
        # print("wxm", data)
        if self.is_output_layer:
            data = data[0]
        self.stat.time_online += time.time() - t
        return data
    
    def send_offline(self, r:np.ndarray) -> None:
        self.send_he(r)
        
    def recv_offline(self) -> None:
        xma = self.recv_he() # f(r_i/m_{i-1}) .* m_i
        xmb = self.recv_he()
        return xma, xmb
        
    def send_online(self, xma, xmb):
        # construct additive share -> send with OT
        xmar, xmbr = self.construct_add_share(xma, xmb) # x_i m_{i-1} - r_i
        ns, nr = self.ots.run_customized(xmar, xmbr, self.ot_callback)
        self.stat.byte_online_send += ns
        self.stat.byte_online_recv += nr
    
    def recv_online(self):
        # receive additive share -> reconstruct
        xma = self.recv_plain() # f(x_i - r_i/m_{i-1}) .* m_i
        xmb = self.recv_plain()
        xma, xmb = self.reconstruct_add_data(xma, xmb) # f(x_i) .* m_i
        return xma, xmb
    
    def set_r(self):
        self.r = gen_add_share(self.ishape)
        
    def construct_add_share(self, data_a, data_b):
        return data_a - self.r, data_b - self.r
    
    def reconstruct_add_data(self, share_a, share_b):
        return share_a + self.pre[0], share_b + self.pre[1]
    
    
class LayerServer(LayerCommon):
    def __init__(self, socket:socket, ishape:tuple, oshape:tuple, layer:torch.nn.Module) -> None:
        super().__init__(socket, ishape, oshape, Pyfhel())
        self.layer = layer
        self.mlast = None
        self.hlast = None
        self.m = None
        #self.m_neg = None
        self.h = None # binary mask
        self.ma, self.mb = None, None
        self.otr = None # oblivious transfer receiver
    
    def setup(self, last_lyr: LayerCommon, m: Union[torch.Tensor, float, int]=None, **kwargs) -> None:
        '''
        Setup the layer's parameters for data transmission protocol.
        Both for receiving ("m", "h" of last layer), and sending ("m", "h", "ma", "mb" of this layer). 
        If "last_lyr" is None, this is the first layer, set "mlast" and "hlast" to 1.
        If "m" is not given (either a tensor or a value), m will be set randomly.
        If "m" is a tensor, there must be "h", "ma", "mb" tensors in the "kwargs" to set the corresponding values.
        
        "ot_nbits" is use in every layer in this protocol, to set up the number of bits for OT.
        Some parameters in "kwargs" can be used in derived classes.
        '''
        t = time.time()
        # set m and h for last layer
        if last_lyr is None:
            self.mlast = torch.ones(self.ishape)
            self.hlast = torch.ones(self.ishape, dtype=torch.bool)
        else:
            assert isinstance(last_lyr, LayerServer)
            assert last_lyr.m.shape == last_lyr.h.shape == self.ishape
            self.mlast = last_lyr.m
            self.hlast = last_lyr.h
        # set m and h for this layer
        if m is None:
            self.m = gen_mul_share(self.oshape)
            m_neg = -gen_mul_share(self.oshape)
            self.h = torch.randint(0, 2, self.oshape, dtype=torch.bool)
            self.ma = merge_by_h(self.m, m_neg, self.h)
            self.mb = merge_by_h(m_neg, self.m, self.h)
        elif isinstance(m, (int, float)):
            self.m = torch.zeros(self.oshape) + m
            # m_neg = self.m
            self.h = torch.ones(self.oshape, dtype=torch.bool)
            self.ma = self.m
            self.mb = self.m
        elif isinstance(m, torch.Tensor):
            assert m.shape == self.oshape
            assert 'h' in kwargs and isinstance(kwargs['h'], torch.Tensor)
            assert 'ma' in kwargs and isinstance(kwargs['ma'], torch.Tensor)
            assert 'mb' in kwargs and isinstance(kwargs['mb'], torch.Tensor)
            self.m = m
            self.h = kwargs['h']
            self.ma = kwargs['ma']
            self.mb = kwargs['mb']
            assert self.h.shape == self.oshape
            # the following check only holds when last layer is not max/min-pooling
            # assert self.ma.shape == self.oshape
            # assert self.mb.shape == self.oshape
        else:
            raise TypeError("m should be a number or a tensor")
        # set oblivious transfer
        if not isinstance(self, LocalLayerServer):
            if 'ot_nbits' not in kwargs:
                nbits = 1024
            else:
                assert isinstance(kwargs['ot_nbits'], int)
                nbits = kwargs['ot_nbits']
            self.otr = comm.ObliviousTransferReceiver(self.socket, nbits)
            self.otr.setup()
        self.stat.time_offline += time.time() - t
    
    def offline(self) -> np.ndarray:
        # return r_i/m_{i-1}
        raise NotImplementedError
    
    def online(self) -> torch.Tensor:
        # return x_i - r_i/m_{i-1}
        raise NotImplementedError

    def recv_offline(self) -> np.ndarray:
        # receive r by HE -> divide m_last
        data = self.recv_he() # r_i
        data = self.reconstruct_mul_data(data, self.mlast) # r_i/m_{i-1}
        return data

    def send_offline(self, data) -> None:
        # send d*ma and d*mb
        wrma = self.construct_mul_share(data, self.ma) # r_i/m_{i-1} .* m_i
        wrmb = self.construct_mul_share(data, self.mb)
        self.send_he(wrma)
        self.send_he(wrmb)

    def run_layer_offline(self, data:torch.Tensor) -> torch.Tensor:
        bias = self.layer.bias
        self.layer.bias = None
        data = self.layer(data)
        self.layer.bias = bias
        return data

    def recv_online(self):
        # receive multiplicative share with OT -> reconstruct
        # TODO: use a bit serializer
        h_bytes = self.hlast.numpy().tobytes()
        data, ns, nr = self.otr.run_customized(h_bytes) # x_i m_{i-1} - r_i
        data = comm.deserialize_torch(data)
        data = self.reconstruct_mul_data(data) # x_i - r_i/m_{i-1}
        self.stat.byte_online_send += ns
        self.stat.byte_online_recv += nr
        return data
    
    def send_online(self, xr):
        # construct multiplicative shares -> send
        xrma = self.construct_mul_share(xr, self.ma) # f(x_i - r_i/m_{i-1}) .* m_i
        xrmb = self.construct_mul_share(xr, self.mb)
        self.send_plain(xrma)
        self.send_plain(xrmb)
    
    def construct_mul_share(self, data, m):
        '''
        Construct multiplicative share for sending online data.
        Multiply m to data.
        '''
        return data*m
    
    def reconstruct_mul_data(self, share, mlast = None):
        '''
        Reconstruct data from the multiplicative share for received online data.
        Divide mlast from share.
        '''
        if mlast is None:
            mlast = self.mlast
        return share / mlast


# local layer specialization

class LocalLayerClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)
    
    def offline(self) -> None:
        return

    def online(self, xm) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

class LocalLayerServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        super().__init__(socket, ishape, oshape, layer)
    
    def offline(self):
        return
    
    def online(self):
        return

