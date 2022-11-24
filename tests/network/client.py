import socket
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import struct
import io
import torch

BF_SZ = 4096
host, port = "localhost", 8000

key = b'abcdefghijklmnop'
iv = b'ABCDEFGHIJKLMNOP'
cipher = Cipher(algorithms.AES(key), modes.CBC(iv))

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((host, port))
    # 1. strings
    s.sendall(b"Hello, world")
    data = s.recv(BF_SZ)
    print(f"1, receive from [{host}]: {data}")
    # 2. numpy array
    d1 = np.array([1, 2, 3, 4, 5], dtype=np.float32)+0.1
    s.sendall(d1.tobytes())
    data = s.recv(BF_SZ)
    d = np.frombuffer(data, dtype=np.float32)
    print(f"2.1, receive from [{host}]: {d}")
    d2 = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    s.sendall(d2.tobytes())
    data = s.recv(BF_SZ)
    d = np.frombuffer(data, dtype=np.int16)
    print(f"2.2, receive from [{host}]: {d}")
    # 3. AES ciphertext
    e = cipher.encryptor()
    d = e.update(b"0123456789012345") + e.finalize()
    s.sendall(d)
    data = s.recv(BF_SZ)
    print(f"3, receive from [{host}]: {data}")
    # 4. big data (multiple packets)
    shape = (20, 50, 10)
    big = np.random.random(shape)
    type_str = big.dtype.str.encode()
    s.sendall(struct.pack('!iii', big.nbytes, len(type_str), len(shape))+type_str+
              struct.pack('!'+'i'*len(shape), *shape))
    s.sendall(big.tobytes())
    sum1 = big.sum()
    data = s.recv(BF_SZ)
    sum2 = struct.unpack('!f', data[:4])[0]
    n = len(data[4:])//4
    shape = struct.unpack('!'+'i'*n, data[4:])
    print(f"4, receive from [{host}]: {sum2}, {shape}. The difference is {sum1-sum2}")
    # 5. torch tensor
    t = torch.rand((20, 20))
    buffer = io.BytesIO()
    torch.save(t, buffer)
    n = buffer.tell()
    buffer.seek(0)
    sum1 = t.sum()
    s.sendall(buffer.read())
    data = s.recv(BF_SZ)
    sum2 = struct.unpack('!f', data[:4])[0]
    print(f"5, receive from [{host}]: {sum2}. The difference is {sum1-sum2}")
    