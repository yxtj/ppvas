from typing import Union
import torch
import socket

from protocol.protocol import Protocol

class ProtocolActBase(Protocol):
    def __init__(self, s: socket.socket, name: Union[str, bytes], shape: tuple[int]):
        super().__init__(s, name)
        self.shape = shape
    
    def client_init(self):
        pass
    
    def server_init(self):
        pass
    
    def activate(self, data:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    