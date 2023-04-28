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
    def setup(self, sshape: tuple, rshape: tuple, mlast: torch.Tensor, **kwargs) -> None:
        '''
        Setup the layer's parameters for data transmission protocol.
        Both for receiving ("m", "h" of last layer), and sending ("m", "h", "ma", "mb" of this layer). 
        If "last_lyr" is None, this is the first layer, set "mlast" and "hlast" to 1.
        If "m" is not given (either a tensor or a value), m will be set randomly.
        If "m" is a tensor, there must be "h", "ma", "mb" tensors in the "kwargs" to set the corresponding values.
        
        "ot_nbits" is use in every layer in this protocol, to set up the number of bits for OT.
        Some parameters in "kwargs" can be used in derived classes.
        '''
        super().setup(sshape, rshape, mlast, **kwargs)
        # set m and h for last layer
        last_lyr = kwargs.get('last_lyr', None)
        m = kwargs.get('m', None)
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
    
    def recv_offline(self) -> torch.Tensor:
        # data = self.recv_he() # r_i
        # data = self.reconstruct_mul_data(data, self.mlast) # r_i/m_{i-1}
        if USE_HE:
            data, nbytes = comm.recv_he_matrix(self.socket, self.he)
        else:
            data, nbytes = comm.recv_torch(self.socket)
        self.stat.byte_offline_recv += nbytes
        return data
        
    def send_offline(self, data: np.ndarray) -> None:
        # send d*ma and d*mb
        # wrma = self.construct_mul_share(data, self.ma) # r_i/m_{i-1} .* m_i
        # wrmb = self.construct_mul_share(data, self.mb)
        # self.send_he(wrma)
        # self.send_he(wrmb)
        if USE_HE:
            self.stat.byte_offline_send += comm.send_he_matrix(self.socket, data, self.he)
        else:
            self.stat.byte_offline_send += comm.send_torch(self.socket, data)
    
    def recv_online(self) -> torch.Tensor:
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