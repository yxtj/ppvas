from layer.base import LayerClient, LayerServer
from comm.util import send_torch, recv_torch

from socket import socket
import torch

class OutputClient(LayerClient):
    def __init__(self, socket: socket, inshape: tuple, outshape: tuple) -> None:
        super().__init__(socket, inshape, outshape)
    
    
class OutputServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple,
                 layer: torch.nn.Module, mlast: torch.Tensor) -> None:
        super().__init__(socket, ishape, oshape, layer, mlast)
    
    def offline(self) -> None:
        data = recv_torch(self.socket)
        data = self.reconstruct_mul_data(data)
        send_torch(self.socket, data)
    
    def online(self) -> None:
        data = recv_torch(self.socket)
        data = self.reconstruct_mul_data(data)
        send_torch(self.socket, data)
    