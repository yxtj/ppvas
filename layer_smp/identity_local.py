from .base import LocalLayerClient, LocalLayerServer

from socket import socket
from typing import Union
import time
import torch
import torch.nn as nn
from Pyfhel import Pyfhel

class LocalIdentityClient(LocalLayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)
        self.layer = nn.ReLU()
    
    def online(self, xm) -> torch.Tensor:
        return xm
    
class LocalIdentityServer(LocalLayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, nn.Identity)
        super().__init__(socket, ishape, oshape, layer)

    def setup(self, m_last: Union[torch.Tensor, float, int],
              m: Union[torch.Tensor, float, int]=None, **kwargs) -> None:
        t = time.time()
        super().setup(m_last, m_last)
        self.stat.time_offline += time.time() - t
        