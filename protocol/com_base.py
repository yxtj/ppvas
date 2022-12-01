import torch

from protocol.protocol import Protocol

class ProtocolCommBase(Protocol):
    
    def server_init(self):
        pass
    
    def client_init(self):
        pass
 
    def send_torch(self, data:torch.Tensor):
        raise NotImplementedError
    
    def recv_torch(self):
        raise NotImplementedError
    