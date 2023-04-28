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
    
    def online(self, xm) -> tuple[torch.Tensor, torch.Tensor]:
        t = time.time()
        # print("xm", xm)
        self.protocol.send_online(xm)
        data = self.protocol.recv_online()
        # print("wxm", data)
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
        self.protocol = ProtocolServer(socket, self.stat, self.he)
        # self.m = None
        # #self.m_neg = None
        # self.h = None # binary mask
        # self.ma, self.mb = None, None
        # self.otr = None # oblivious transfer receiver
    
    def setup(self, last_lyr: LayerCommon, m: Union[torch.Tensor, float, int]=None, **kwargs) -> None:
        t = time.time()
        mlast = last_lyr.m if last_lyr else None
        self.protocol.setup(self.ishape, self.oshape, mlast, m=m)
        self.stat.time_offline += time.time() - t
    
    def offline(self) -> np.ndarray:
        # return r_i/m_{i-1}
        raise NotImplementedError
    
    def online(self) -> torch.Tensor:
        # return x_i - r_i/m_{i-1}
        raise NotImplementedError

    def recv_offline(self) -> np.ndarray:
        # receive r by HE -> divide m_last
        data = self.protocol.recv_offline()
        return data

    def send_offline(self, data) -> None:
        self.protocol.send_offline(data)

    def run_layer_offline(self, data:torch.Tensor) -> torch.Tensor:
        bias = self.layer.bias
        self.layer.bias = None
        data = self.layer(data)
        self.layer.bias = bias
        return data

    def recv_online(self):
        data = self.protocol.recv_online()
        return data
    
    def send_online(self, xr):
        self.protocol.send_online(xr)


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
    
    def offline(self):
        return
    
    def online(self):
        return

