import torch

from protocol.act_base import ProtocolActBase

class ProtocolActPlain(ProtocolActBase):
    def __init__(self, s, shape):
        super().__init__(s, 'act-plain', shape)
        
    def client_init(self):
        pass
    
    def server_init(self):
        pass
    
    def server_send_prepare(self, data:torch.Tensor):
        return data
    
    def activate(self, data:torch.Tensor) -> torch.Tensor:
        return torch.relu(data)
    
    def client_send_prepare(self, data:torch.Tensor):
        return data
    