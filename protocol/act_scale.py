import torch

from protocol.act_base import ProtocolActBase
from protocol.util import send_torch, recv_torch

class ProtocolActScale(ProtocolActBase):
    def __init__(self, s, shape):
        super().__init__(s, 'act-scale', shape)
        
    def client_init(self, offset:torch.Tensor, next_share:torch.Tensor):
        self.offset = offset
        self.next_share = next_share
    
    def server_init(self):
        self.m = torch.rand(self.shape)
    
    def server_send(self, data:torch.Tensor):
        data *= self.m
        send_torch(self.s, data)
        
    def client_recv(self):
        data = recv_torch(self.s)
        return data
    
    def activate(self, data:torch.Tensor) -> torch.Tensor:
        return torch.relu(data + self.offset)
    
    def client_send(self, data:torch.Tensor):
        send_torch(self.s, data - self.next_share)
    
    def server_recv(self):
        data = recv_torch(self.s)
        return data
    
    