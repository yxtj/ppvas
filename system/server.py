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
        self.layers, self.shortcuts = util.make_server_model(socket, model, inshape)
        self.to_buffer = [v for k,v in self.shortcuts.items()]
    
    def offline(self):
        buffer = {}
        mlast = 1
        for i, lyr in enumerate(self.layers):
            name = lyr.__class__.__name__
            print('  offline {}: {}(inshape={}, outshape={}) ...'.format(i, name, lyr.ishape, lyr.oshape))
            # setup
            if i in self.shortcuts:
                # assert isinstance(lyr, layer.shortcut.ShortCutServer)
                idx = self.shortcuts[i]
                m_other = self.layers[idx].m
                lyr.setup(mlast, m_other)
            else:
                lyr.setup(mlast)
            mlast = lyr.m
            # offline
            if i in self.shortcuts:
                idx = self.shortcuts[i]
                t = lyr.offline(buffer[idx])
            else:
                t = lyr.offline()
            if i in self.to_buffer:
                buffer[i] = t
            
    def online(self):
        buffer = {}
        for i, lyr in enumerate(self.layers):
            if i in self.shortcuts:
                idx = self.shortcuts[i]
                t = lyr.online(buffer[idx])
            else:
                t = lyr.online()
            if i in self.to_buffer:
                buffer[i] = t
        