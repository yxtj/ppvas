import socket
import struct

class Protocol():
    def __init__(self, s:socket.socket, name:bytes):
        self.s = s
        self.name = name
        self.nameb = bytes(name.encode('utf-8'))
    
    def send_name(self):
        self.s.send(struct.pack('!is', len(self.nameb), self.nameb))
    
    def check_name(self):
        buff = self.s.recv(4)
        n = struct.unpack('!i', buff[:4])[0]
        buff = self.s.recv(n)
        assert buff == self.nameb, 'Protocol name mismatch'
    
    def client_side(self):
        pass
    
    def server_side(self):
        pass
