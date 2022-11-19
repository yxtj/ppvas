import socketserver
import time, os
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import struct

BF_SZ = 4096

key = b'abcdefghijklmnop'
iv = b'ABCDEFGHIJKLMNOP'
cipher = Cipher(algorithms.AES(key), modes.CBC(iv))

class MyTCPHandler(socketserver.BaseRequestHandler):
    
    def setup(self) -> None:
        super().setup()
        self.cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """
    def handle(self):
        # self.request is the TCP socket connected to the client
        # 1. strings
        data = self.request.recv(BF_SZ)
        #print("{} wrote:".format(self.client_address[0]))
        print(f"1, receive from [{self.client_address}]: {data}")
        # just send back the same data, but upper-cased
        self.request.sendall(data.upper())
        # 2. numpy array
        data = self.request.recv(BF_SZ)
        d = np.frombuffer(data, dtype=np.float32) # result of frombuffer is read-only when the data is a bytes string
        print(f"2.1, receive from [{self.client_address}]: {d}")
        d = d * 2
        self.request.sendall(d.tobytes())
        data = self.request.recv(BF_SZ)
        d = np.frombuffer(data, dtype=np.int16)
        print(f"2.2, receive from [{self.client_address}]: {d}")
        d = d + 1
        self.request.sendall(d.tobytes())
        # 3. AES ciphertext
        data = self.request.recv(BF_SZ)
        d = self.cipher.decryptor()
        r = d.update(data) + d.finalize()
        print(f"3, receive from [{self.client_address}]: {r}")
        self.request.sendall(r)
        # 4. big data (multiple packets)
        buffer = []
        chunck = self.request.recv(BF_SZ)
        n, len_type_str, num_shape = struct.unpack('!iii', chunck[:12])
        header_len = 12 + len_type_str + num_shape*4
        type_str = chunck[12:12+len_type_str].decode()
        shape = struct.unpack('!'+'i'*num_shape, chunck[12+len_type_str:12+len_type_str+num_shape*4])
        print(f"4.0, receive from [{self.client_address}]: n={n}, type={type_str}, shape={shape}")
        if len(chunck) > header_len:
            print(f"4.x, receive from [{self.client_address}]: {len(buffer)+1}-th len={len(chunck)-header_len} left={n}")
            buffer.append(chunck[header_len:])
            n -= len(chunck) - header_len
        while n > 0:
            chunck = self.request.recv(BF_SZ)
            print(f"4.x, receive from [{self.client_address}]: {len(buffer)+1}-th len={len(chunck)} left={n}")
            n -= len(chunck)
            buffer.append(chunck)
        data = b''.join(buffer)
        d = np.frombuffer(data, dtype=type_str).reshape(shape)
        s = d.sum()
        print(f"4, receive from [{self.client_address}]: {len(data)} {d.shape} {s}")
        self.request.sendall(struct.pack('!f', s)+struct.pack('!'+'i'*len(d.shape), *d.shape))
        print('close connection')
        
host, port = "localhost", 8000
with socketserver.TCPServer((host, port), MyTCPHandler) as server:
    print('start server')
    server.serve_forever()
    