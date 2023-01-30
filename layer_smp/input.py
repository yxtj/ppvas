from .base import LocalLayerClient, LocalLayerServer

from socket import socket
from typing import Union
import time
import torch
import torch.nn as nn
from Pyfhel import Pyfhel

class InputClient(LocalLayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)
    
    def online(self, x) -> torch.Tensor:
        return x
    
class InputServer(LocalLayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, nn.Identity)
        super().__init__(socket, ishape, oshape, layer)

    def setup(self, last_lyr: LayerServer, m: Union[torch.Tensor, float, int]=None, **kwargs) -> None:
        t = time.time()
        super().setup(last_lyr, m)
        self.stat.time_offline += time.time() - t
        