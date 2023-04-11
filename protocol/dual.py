from .ptobase import ProBaseServer, ProBaseClient

import torch
import numpy as np
import comm
from setting import USE_HE

__ALL__ = ['ProtocolClient', 'ProtocolServer']

def merge_by_h(A: torch.Tensor, B: torch.Tensor, H: torch.Tensor):
    return A * H + B * ~H
class ProtocolClient(ProBaseClient):
    def __init__(self, s, stat, he):
        super().__init__(s, stat, he)
        self.is_input_layer = False
        self.is_output_layer = False
    
    def setup(self, sshape: tuple, rshape: tuple, **kwargs) -> None: 
        super().setup(sshape, rshape)
        # set additive share
        self.set_r()
        # set input and output attribute
        if 'is_input_layer' in kwargs and isinstance(kwargs['is_input_layer'], bool):
            self.is_input_layer = kwargs['is_input_layer']
        if 'is_output_layer' in kwargs and isinstance(kwargs['is_output_layer'], bool):
            self.is_output_layer = kwargs['is_output_layer']
        if 'ot_nbits' in kwargs and isinstance(kwargs['ot_nbits'], int):
            ot_nbits = kwargs['ot_nbits']
        self.ots = comm.ObliviousTransferSender(self.socket, ot_nbits)
        self.ots.setup()
    
    def send_offline(self, data: torch.Tensor) -> None:
        nbytes = self._basic_he_send_(data)
        self.stat.byte_offline_send += nbytes

    def recv_offline(self) -> torch.Tensor:
        # f(r_i/m_{i-1}) .* m_i
        xma, nba = self._basic_he_recv_()
        xmb, nbb = self._basic_he_recv_()
        self.stat.byte_offline_recv += nba + nbb
        self.pre = (xma, xmb)
        return xma, xmb

    def ot_callback(self, xa, xb, h_bytes):
        # TODO: use a bit serizlizer
        h = np.frombuffer(bytes([i & 1 for i in h_bytes]), dtype=np.bool)
        h = torch.from_numpy(h).reshape(xa.shape)
        o = merge_by_h(xa, xb, h)
        o = comm.serialize_torch(o)
        return o
    
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
