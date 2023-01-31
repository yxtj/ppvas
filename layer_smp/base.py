from socket import socket
from typing import Union
import time
import torch
# import numpy as np
from Pyfhel import Pyfhel

# import comm
from layer_basic import LayerCommon, gen_add_share, gen_mul_share

__all__ = ['LayerClient', 'LayerServer', 'LocalLayerClient', 'LocalLayerServer']


class LayerClient(LayerCommon):
    def __init__(self, socket:socket, ishape:tuple, oshape:tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)
        self.r = None
        self.pre = None
    
    def setup(self, **kwargs):
        return
    
    def offline(self) -> None:
        t = time.time()
        self.set_r()
        # print("r", self.r)
        self.send_he(self.r)
        data = self.recv_he()
        self.pre = data
        # print("pre", self.pre)
        self.stat.time_offline += time.time() - t
    
    def online(self, xm) -> torch.Tensor:
        t = time.time()
        # print("xm", xm)
        data = self.construct_add_share(xm)
        # print("xm-r", data)
        self.send_plain(data)
        data = self.recv_plain()
        # print("w(x-r/m)m", data)
        data = self.reconstruct_add_data(data)
        # print("wxm", data)
        self.stat.time_online += time.time() - t
        return data
    
    def set_r(self):
        self.r = gen_add_share(self.ishape)
        # self.r = torch.zeros(self.ishape)
        # self.r = torch.zeros(self.ishape) + np.random.randint(1, 6)*0.1
    
    def construct_add_share(self, data):
        return data - self.r
    
    def reconstruct_add_data(self, share):
        return share + self.pre
    
    
class LayerServer(LayerCommon):
    def __init__(self, socket:socket, ishape:tuple, oshape:tuple, layer:torch.nn.Module) -> None:
        super().__init__(socket, ishape, oshape, Pyfhel())
        self.layer = layer
        self.mlast = None
        self.m = None
    
    def setup(self, last_lyr: LayerCommon, m: Union[torch.Tensor, float, int]=None, **kwargs) -> None:
        '''
        Setup the layer's factors "mlast" and "m". In special cases, mlast may be a constant like 1.
        If m is not given (either a tensor or a value), m will be set randomly.
        For some layers, parameters in kwargs can be used in derivide.
        1. "m_other" is used in shortcut layers (compute x_{i-1} + x_j), to set up the "mlast" for the layer j.
        '''
        # set m for last layer
        if last_lyr is None:
            self.mlast = torch.ones(self.ishape)
        else:
            assert isinstance(last_lyr, LayerServer)
            self.mlast = last_lyr.m
        # set m for this layer
        if m is None:
            self.set_m_positive()
        else:
            if isinstance(m, (int, float)):
                self.m = torch.zeros(self.oshape) + m
            elif isinstance(m, torch.Tensor):
                assert m.shape == self.oshape
                self.m = m
            else:
                raise TypeError("m should be a number or a tensor")
    
    def offline(self) -> torch.Tensor:
        raise NotImplementedError
    
    def online(self) -> torch.Tensor:
        raise NotImplementedError

    def run_layer_offline(self, data:torch.Tensor) -> torch.Tensor:
        bias = self.layer.bias
        self.layer.bias = None
        data = self.layer(data)
        self.layer.bias = bias
        return data

    def set_m_one(self) -> None:
        self.m = torch.ones(self.oshape)
    
    def set_m_positive(self) -> None:
        self.m = gen_mul_share(self.oshape)
        # self.m = torch.ones(self.oshape)
    
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


# local layer specialization

class LocalLayerClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)
    
    def offline(self) -> None:
        return

    def online(self, xm) -> torch.Tensor:
        raise NotImplementedError

class LocalLayerServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        super().__init__(socket, ishape, oshape, layer)
    
    def offline(self) -> None:
        return
    
    def online(self) -> None:
        return

