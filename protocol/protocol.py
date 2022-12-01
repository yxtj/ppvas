import socket
import struct
from typing import Union

class Protocol():
    def __init__(self, s:socket.socket, name:Union[str, bytes]):
        self.s = s
        if isinstance(name, str):
            name = name.encode('utf-8')
        self.name = name
    
    def send_name(self):
        self.s.sendall(struct.pack('!is', len(self.name), self.name))
    
    def check_name(self):
        buff = self.s.recv(4)
        n = struct.unpack('!i', buff[:4])[0]
        buff = self.s.recv(n)
        assert buff == self.name, 'Protocol name mismatch'
    
    def get_name(self):
        return self.name
    
    def client_init(self):
        pass
    
    def server_init(self):
        pass
    