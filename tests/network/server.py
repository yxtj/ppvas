import socketserver
import socket
import time, os
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import struct
import io
import torch

BF_SZ = 4096

key = b'abcdefghijklmnop'
iv = b'ABCDEFGHIJKLMNOP'
cipher = Cipher(algorithms.AES(key), modes.CBC(iv))

def func(conn, addr):
    # 1. strings
    data = conn.recv(BF_SZ)
    #print("{} wrote:".format(addr[0]))
    print(f"1, receive from [{addr}]: {data}")
    # just send back the same data, but upper-cased
    conn.sendall(data.upper())
    # 2. numpy array
    data = conn.recv(BF_SZ)
    d = np.frombuffer(data, dtype=np.float32) # result of frombuffer is read-only when the data is a bytes string
    print(f"2.1, receive from [{addr}]: {d}")
    d = d * 2
    conn.sendall(d.tobytes())
    data = conn.recv(BF_SZ)
    d = np.frombuffer(data, dtype=np.int16)
    print(f"2.2, receive from [{addr}]: {d}")
    d = d + 1
    conn.sendall(d.tobytes())
    # 3. AES ciphertext
    data = conn.recv(BF_SZ)
    d = cipher.decryptor()
    r = d.update(data) + d.finalize()
    print(f"3, receive from [{addr}]: {r}")
    conn.sendall(r)
    # 4. big data (multiple packets)
    buffer = []
    chunck = conn.recv(BF_SZ)
    n, len_type_str, num_shape = struct.unpack('!iii', chunck[:12])
    header_len = 12 + len_type_str + num_shape*4
    type_str = chunck[12:12+len_type_str].decode()
    shape = struct.unpack('!'+'i'*num_shape, chunck[12+len_type_str:12+len_type_str+num_shape*4])
    print(f"4.0, receive from [{addr}]: n={n}, type={type_str}, shape={shape}")
    if len(chunck) > header_len:
        print(f"4.x, receive from [{addr}]: {len(buffer)+1}-th len={len(chunck)-header_len} left={n}")
        buffer.append(chunck[header_len:])
        n -= len(chunck) - header_len
    while n > 0:
        chunck = conn.recv(BF_SZ)
        print(f"4.x, receive from [{addr}]: {len(buffer)+1}-th len={len(chunck)} left={n}")
        n -= len(chunck)
        buffer.append(chunck)
    data = b''.join(buffer)
    d = np.frombuffer(data, dtype=type_str).reshape(shape)
    s = d.sum()
    print(f"4, receive from [{addr}]: {len(data)} {d.shape} {s}")
    conn.sendall(struct.pack('!f', s)+struct.pack('!'+'i'*len(d.shape), *d.shape))
    # 5. torch tensor
    data = conn.recv(BF_SZ)
    d = torch.load(io.BytesIO(data))
    s = d.sum()
    print(f"5, receive from [{addr}]: {len(data)} {d.shape} {s}")
    conn.sendall(struct.pack('!f', s))
    print('close connection')

class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """
    def handle(self):
        # self.request is the TCP socket connected to the client
        conn = self.request
        addr = self.client_address
        func(conn, addr)
        
        
host, port = "localhost", 8000

def method1():
    with socketserver.TCPServer((host, port), MyTCPHandler) as server:
        print('start server')
        server.serve_forever()
        
def method2():
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            s.listen()
            conn, addr = s.accept()
            with conn:
                func(conn, addr)

def method3():
    while True:
        s = socket.create_server((host, port), family=socket.AF_INET)
        conn, addr = s.accept()
        with conn:
            func(conn, addr)
    

# method1()
# method2()
method3()
