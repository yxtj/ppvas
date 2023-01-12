import socket
import numpy as np
import struct
import io
import torch
from Pyfhel import Pyfhel, PyCtxt

def serialize_numpy_meta(data:np.ndarray) -> bytes:
    type_str = data.dtype.str.encode()
    shape = data.shape
    r = struct.pack('!iii', data.nbytes, len(type_str), len(shape))+type_str+struct.pack('!'+'i'*len(shape), *shape)
    return r

def _recv_big_data_(s:socket.socket, n:int, left:bytes=None, buf_sz:int=4096) -> bytes:
    buffer = [left] if left is not None else []
    while n > 0:
        t = s.recv(min(buf_sz, n))
        n -= len(t)
        buffer.append(t)
    data = b''.join(buffer)
    return data

# byte chunk

def send_chunk(s:socket.socket, data:bytes) -> int:
    s.sendall(struct.pack('!i', len(data))+data)
    return len(data)+4

def recv_chunk(s:socket.socket, buf_sz:int=4096) -> bytes:
    data = s.recv(buf_sz)
    n, = struct.unpack('!i', data[:4])
    return _recv_big_data_(s, n, data[4:], buf_sz)

# numpy
    
def send_numpy(s:socket.socket, data:np.ndarray) -> int:
    type_str = data.dtype.str.encode()
    shape = data.shape
    s.sendall(struct.pack('!iii', data.nbytes, len(type_str), len(shape))+type_str+
              struct.pack('!'+'i'*len(shape), *shape))
    data = data.tobytes()
    s.sendall(data)
    return 12 + len(type_str) + len(shape)*4 + len(data)
    
def recv_numpy(s:socket.socket, buf_sz:int=4096) -> tuple(np.ndarray, int):
    data = s.recv(buf_sz)
    nbytes, len_type, len_shape = struct.unpack('!iii', data[:12])
    header_len = 12 + len_type + len_shape*4
    type_str = data[12:12+len_type].decode()
    shape = struct.unpack('!'+'i'*len_shape, data[12+len_type:12+len_type+4*len_shape])
    buffer = _recv_big_data_(s, nbytes, data[header_len:])
    result = np.frombuffer(buffer, dtype=type_str).reshape(shape)
    return result, header_len + nbytes

# tensor

def send_torch(s:socket.socket, data:torch.Tensor) -> int:
    buffer = io.BytesIO()
    torch.save(data, buffer)
    n = buffer.tell()
    buffer.seek(0)
    s.send(struct.pack('!i', n))
    s.sendall(buffer.read())
    return 4 + n
    
def recv_torch(s:socket.socket, buf_sz:int=4096) -> tuple(torch.Tensor, int):
    data = s.recv(buf_sz)
    nbytes, = struct.unpack('!i', data[:4])
    buffer = _recv_big_data_(s, nbytes, data[4:])
    result = torch.load(io.BytesIO(buffer))
    return result, 4 + nbytes

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

# HE ciphertext

def send_ciphertext(s:socket.socket, data:PyCtxt) -> int:
    return send_chunk(s, data.to_bytes())

def recv_ciphertext(s:socket.socket, he:Pyfhel, buf_sz:int=4096) -> tuple(PyCtxt, int):
    data = recv_chunk(s, buf_sz)
    res = PyCtxt(pyfhel=he, bytestring=data)
    return res, len(data) + 4
