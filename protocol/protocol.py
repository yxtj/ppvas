import socket

class Protocol():
    def __init__(self, s:socket.socket, name:bytes):
        self.s = s
        self.name = name
    
    def client_side(self):
        pass
    
    def server_side(self):
        pass
