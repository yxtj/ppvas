import socket
import torch
import torch.nn as nn

from model import util
import layer

class Server():
    def __init__(self, socket: socket.socket, model: nn.Module, inshape: tuple):
        self.socket = socket
        # model
        self.model = model
        self.inshape = inshape
        # layers
        self.layers, self.to_buffer = util.make_server_model(socket, model, inshape)
        
    def offline(self):
        buffer = {}
        for i, lyr in enumerate(self.layers):
            print('  offline {}: {}(inshape={}, outshape={}) ...'.format(
                i, lyr.__class__.__name__, lyr.ishape, lyr.oshape))
            if i in self.to_buffer:
                if isinstance(lyr, layer.shortcut.ShortCutServer):
                    t = lyr.offline(buffer[i + lyr.other_offset])
                else:
                    t = lyr.offline()
                buffer[i] = t
            else:
                lyr.offline()
            
    def online(self):
        buffer = {}
        for i, lyr in enumerate(self.layers):
            if i in self.to_buffer:
                if isinstance(lyr, layer.shortcut.ShortCutServer):
                    t = lyr.online(buffer[i + lyr.other_offset])
                else:
                    t = lyr.online()
                buffer[i] = t
            lyr.online()
        