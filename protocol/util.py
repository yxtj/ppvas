import socket
import numpy as np
import struct
import io
import torch

def serialize_numpy_meta(data:np.ndarray):
    type_str = data.dtype.str.encode()
    shape = data.shape
    r = struct.pack('!iii', data.nbytes, len(type_str), len(shape))+type_str+struct.pack('!'+'i'*len(shape), *shape)
    return r

# numpy
    
def send_numpy(s:socket.socket, data:np.ndarray):
    type_str = data.dtype.str.encode()
    shape = data.shape
    s.sendall(struct.pack('!iii', data.nbytes, len(type_str), len(shape))+type_str+
              struct.pack('!'+'i'*len(shape), *shape))
    s.sendall(data.tobytes())
    
def recv_numpy(s:socket.socket, buf_sz:int=4096) -> np.ndarray:
    data = s.recv(buf_sz)
    nbytes, len_type, len_shape = struct.unpack('!iii', data[:12])
    header_len = 12 + len_type + len_shape*4
    type_str = data[12:12+len_type].decode()
    shape = struct.unpack('!'+'i'*len_shape, data[12+len_type:12+len_type+4*len_shape])
    buffer = []
    if len(chunck) > header_len:
            buffer.append(chunck[header_len:])
            nbytes -= len(chunck) - header_len
    while nbytes > 0:
        chunck = s.recv(buf_sz)
        nbytes -= len(chunck)
        buffer.append(chunck)
    buffer = b''.join(buffer)
    result = np.frombuffer(buffer, dtype=type_str).reshape(shape)
    return result

# tensor

def send_torch(s:socket.socket, data:torch.Tensor):
    buffer = io.BytesIO()
    torch.save(data, buffer)
    n = buffer.tell()
    buffer.seek(0)
    s.sendall(buffer.read())
    
def recv_torch(s:socket.socket, buf_sz:int=4096) -> torch.Tensor:
    data = s.recv(buf_sz)
    nbytes, = struct.unpack('!i', data[:4])
    buffer = []
    if len(chunck) > 4:
            buffer.append(chunck[4:])
            nbytes -= len(chunck) - 4
    while nbytes > 0:
        chunck = s.recv(buf_sz)
        nbytes -= len(chunck)
        buffer.append(chunck)
    buffer = b''.join(buffer)
    result = torch.load(io.BytesIO(buffer))
    return result

# shape tuple

def send_shape(s:socket.socket, shape:tuple[int]):
    s.sendall(struct.pack('!i', len(shape)))
    s.sendall(struct.pack('!'+'i'*len(shape), *shape))
    
def recv_shape(s:socket.socket) -> tuple[int]:
    data = s.recv(4)
    n, = struct.unpack('!i', data)
    data = s.recv(n*4)
    shape = struct.unpack('!'+'i'*n, data)
    return shape
