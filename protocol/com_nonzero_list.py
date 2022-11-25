from protocol.protocol import Protocol
import struct
import torch
from protocol.util import send_torch, recv_torch, send_shape, recv_shape

class PtlCommNonZeroList(Protocol):
    def __init__(self, s):
        super().__init__(s, 'comm-nonzero-list')

    def send_nz_lise(self, data:torch.Tensor):
        send_shape(self.s, data.shape)
        nz_idx = torch.nonzero(data).type(torch.int16)
        n = len(nz_idx)
        self.s.send(struct.pack('!i', n))
        send_torch(self.s, nz_idx)
        nz_idx_tuple = [nz_idx[:,i] for i in range(nz_idx.shape[1])]
        send_torch(self.s, data[nz_idx_tuple])

    def recv_nz_list(self):
        shape = recv_shape(self.s)
        buff = self.s.recv(4)
        n = struct.unpack('!i', buff[:4])[0]
        nz_idx = recv_torch(self.s)
        nz_idx_tuple = [nz_idx[:,i] for i in range(nz_idx.shape[1])]
        data = recv_torch(self.s)
        res = torch.zeros(data.shape)
        res[nz_idx_tuple] = data
        return res

    def client_side(self, data:torch.Tensor=None):
        if data is not None:
            self.send_name()
            self.send_nz_lise(data)
        else:
            self.check_name()
            return self.recv_nz_list()

    def server_side(self, data:torch.Tensor=None):
        if data is not None:
            self.send_name()
            self.send_nz_lise(data)
        else:
            self.check_name()
            return self.recv_nz_list()
        