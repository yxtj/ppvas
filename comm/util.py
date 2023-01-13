import socket
import numpy as np
import struct
import io
import torch
from Pyfhel import Pyfhel, PyCtxt

def serialize_numpy_meta(data:np.ndarray) -> bytes:
    type_str = data.dtype.str.encode()
    shape = data.shape
    r = struct.pack('!iii', data.nbytes, len(type_str), len(shape)) + \
        type_str + struct.pack('!'+'i'*len(shape), *shape)
    return r

def deserialize_numpy_meta(data:bytes) -> tuple[int, int, str, tuple[int]]:
    nbytes, type_len, shape_len = struct.unpack('!iii', data[:12])
    header_len = 12 + type_len + shape_len*4
    type_str = data[12 : 12+type_len].decode()
    shape = struct.unpack('!'+'i'*shape_len, data[12+type_len : header_len])
    return header_len, nbytes, type_str, shape

def _recv_big_data_(s:socket.socket, n:int, left:bytes=None, buf_sz:int=4096) -> tuple[bytes, bytes]:
    if left is not None and len(left) >= n:
        return left[:n], left[n:]
    buffer = [left] if left is not None else []
    while n > 0:
        t = s.recv(min(buf_sz, n))
        n -= len(t)
        buffer.append(t)
    data = b''.join(buffer)
    return data, b''

# byte chunk

def send_chunk(s:socket.socket, data:bytes) -> int:
    s.sendall(struct.pack('!i', len(data))+data)
    return len(data)+4

def recv_chunk(s:socket.socket, buf_sz:int=4096) -> bytes:
    data = s.recv(buf_sz)
    n, = struct.unpack('!i', data[:4])
    return _recv_big_data_(s, n, data[4:], buf_sz)[0]

# numpy
    
def send_numpy(s:socket.socket, data:np.ndarray) -> int:
    b_meta = serialize_numpy_meta(data)
    s.sendall(b_meta)
    data = data.tobytes()
    s.sendall(data)
    return len(b_meta) + len(data)
    
def recv_numpy(s:socket.socket, buf_sz:int=4096) -> tuple[np.ndarray, int]:
    data = s.recv(buf_sz)
    header_len, nbytes, type_str, shape = deserialize_numpy_meta(data)
    buffer = _recv_big_data_(s, nbytes, data[header_len:])[0]
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
    
def recv_torch(s:socket.socket, buf_sz:int=4096) -> tuple[torch.Tensor, int]:
    data = s.recv(buf_sz)
    nbytes, = struct.unpack('!i', data[:4])
    buffer = _recv_big_data_(s, nbytes, data[4:])[0]
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

def recv_ciphertext(s:socket.socket, he:Pyfhel, buf_sz:int=4096) -> tuple[PyCtxt, int]:
    data = recv_chunk(s, buf_sz)
    res = PyCtxt(pyfhel=he, bytestring=data)
    return res, len(data) + 4


def send_he_matrix(s:socket.socket, data:np.ndarray, he:Pyfhel) -> int:
    b_ctx = he.to_bytes_context()
    meta = serialize_numpy_meta(data)
    d = data.flatten()
    header = struct.pack('!i', len(b_ctx)) + b_ctx + meta + struct.pack('!i', len(d[0].to_bytes()))
    s.sendall(header)
    n = 4 + len(b_ctx) + len(meta) + 4
    for i in range(data.size):
        n += s.sendall(d[i].to_bytes())
    return n

def recv_he_matrix(s:socket.socket, he:Pyfhel, buf_sz:int=4096) -> tuple[np.ndarray, int]:
    header_len = struct.unpack('!i', s.recv(4))[0]
    data = s.recv(header_len)
    recv_chunk(s, buf_sz)
    ctx_len = struct.unpack('!i', data[:4])[0]
    ctx = data[4:4+ctx_len]
    meta_len, nbytes, type_str, shape = deserialize_numpy_meta(data[4+ctx_len:])
    ct_len = struct.unpack('!i', data[4+ctx_len+meta_len:])[0]
    # update HE context
    he.from_bytes_context(ctx)
    # parse ciphertexts
    res = np.empty(shape, dtype=object)
    r = res.flatten()
    size = np.prod(shape)
    for i in range(size):
        b = _recv_big_data_(s, ct_len)[0]
        ct = PyCtxt(pyfhel=he, bytestring=b)
        r[i] = ct
    return res, 4 + header_len + nbytes
