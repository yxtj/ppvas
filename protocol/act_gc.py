import torch
from .act_base import ProtocolActBase
from mpc.gc import GarbledCircuit

class PtlActGarbledCircuit(ProtocolActBase):
    def __init__(self, s):
        super().__init__(s, 'act-gc')
    
    def client_init(self):
        pass
    
    def server_init(self):
        pass
    
    def server_send_prepare(self, data:torch.Tensor) -> torch.Tensor:
        return data
    
    def activate(self, data:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def client_send_prepare(self, data:torch.Tensor) -> torch.Tensor:
        return data