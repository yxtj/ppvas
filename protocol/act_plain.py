import torch

from protocol.act_base import ProtocolActBase
from protocol.util import send_torch, recv_torch

class ProtocolActPlain(ProtocolActBase):
    def __init__(self, s, shape):
        super().__init__(s, 'act-plain', shape)
        
    def client_init(self):
        pass
    
    def server_init(self):
        pass
    
    def server_send(self, data:torch.Tensor):
        send_torch(self.s, data)
        
    def client_recv(self):
        data = recv_torch(self.s)
        return data
    
    def activate(self, data:torch.Tensor) -> torch.Tensor:
        return torch.relu(data)
    
    def client_send(self, data:torch.Tensor):
        send_torch(self.s, data)
    
    def server_recv(self):
        data = recv_torch(self.s)
        return data
    