from protocol.protocol import Protocol
import struct
import torch
from protocol.util import send_torch, recv_torch

class PtlCommSecretShare(Protocol):
    def __init__(self, s):
        super().__init__(s, 'comm-share')

    def client_side(self):
        self.check_name()
        buff = self.s.recv(4)
        lid = struct.unpack('!i', buff[:4])[0]
        t = recv_torch(self.s)
        return lid, t

    def server_side(self, lid:int, share:torch.Tensor):
        self.send_name()
        self.s.sendall(struct.pack('!i', lid))
        send_torch(self.s, share)
        
