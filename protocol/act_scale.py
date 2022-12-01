import torch

from protocol.act_base import ProtocolActBase

class ProtocolActScale(ProtocolActBase):
    def __init__(self, s, shape):
        super().__init__(s, 'act-scale', shape)
        
    def client_init(self, offset:torch.Tensor, next_share:torch.Tensor):
        self.offset = offset
        self.next_share = next_share
    
    def server_init(self):
        self.m = torch.rand(self.shape)
    
    def server_send_prepare(self, data:torch.Tensor) -> torch.Tensor:
        return data * self.m
    
    def activate(self, data:torch.Tensor) -> torch.Tensor:
        return torch.relu(data + self.offset)
    
    def client_send_prepare(self, data:torch.Tensor) -> torch.Tensor:
        return data - self.next_share
    