from .rsa import RSA
from socket import socket
import struct
import Crypto.Util.strxor
# import Crypto.Random

'''
1-of-2 Oblivious Transfer Protocol.

          Client                                Server
have data: m0, m1                have: b (0/1) indicating which data to require

# RSA setup phase (only once)
generate RSK key (n, e, d)      |
send public key: n, e           =>          receive: n, e

# indicator preparation phase
generate random byte: k0, k1    =>          receive: k0, k1
                                |   generate random mask: h of length m
                                |   compute: c = rsa.encrypt(h) XOR k_b
receive c                       <=          send c
h0' = rsa.decrypt(c XOR k0)     |
h1' = rsa.decrypt(c XOR k1)     |

# data transfer phase
send: m0' = m0 XOR h0'          =>          receive: m0'
      m1' = m1 XOR h1'          =>          receive: m1'
                                |           pick m_b'
                                |           m_b = m_b' XOR h
'''

def xor_mask_char(data_bytes:bytes, mask:int) -> bytes:
    assert 0<= mask <= 255
    return Crypto.Util.strxor.strxor_c(data_bytes, mask)

def xor_mask_bytes(data_bytes:bytes, mask_bytes:bytes) -> bytes:
    return Crypto.Util.strxor.strxor(data_bytes, mask_bytes)

def mask_data(data_bytes:bytes, mask:bytes) -> bytes:
    nbyte = len(mask)
    n = len(data_bytes)
    m, r = divmod(n, nbyte)
    buffer = [xor_mask_bytes(data_bytes[i*nbyte:(i+1)*nbyte], mask) for i in range(m)]
    if r > 0:
        buffer.append(xor_mask_bytes(data_bytes[-r:], mask[:r]))
    return b''.join(buffer)


class ObliviousTransferClient():
    def __init__(self, socket:socket, nbits:int=2048) -> None:
        self.socket = socket
        self.nbits = nbits
        self.nbyte = nbits//8 + (1 if nbits % 8 > 0 else 0)
        self.rsa = RSA(nbits)
        self.mask_len = self.nbyte - 1
        
    def setup(self):
        self.rsa.setup()
        print(f'Client: n={self.rsa.n}\nd ={self.rsa.d}')
        self.socket.sendall(self.rsa.n.to_bytes(self.nbyte, 'big'))
        self.socket.sendall(self.rsa.e.to_bytes(self.nbyte, 'big'))
    
    def run(self, x0:bytes, x1:bytes) -> tuple[int, int]:
        '''
        Oblivious transfer for sending data x0 or x1.
        Return the number of bytes sent and received.
        '''
        # indicator preparation phase
        k0, k1 = self.random_pair()
        data = struct.pack('!BB', k0, k1)
        self.socket.sendall(data)
        c = self.socket.recv(self.nbyte)
        h0 = self.decrypt_with_mask(c, k0)
        h1 = self.decrypt_with_mask(c, k1)
        # data transfer phase
        cnt0 = self.send_data(x0, h0)
        cnt1 = self.send_data(x1, h1)
        return 2 + cnt0 + cnt1, self.nbyte
    
    # local functions
    
    def random_pair(self) -> tuple[int, int]:
        mask1 = Crypto.Random.random.randint(0, 255)
        mask2 = Crypto.Random.random.randint(0, 255)
        return mask1, mask2

    def decrypt_with_mask(self, data_bytes:bytes, mask:int) -> bytes:
        d = xor_mask_char(data_bytes, mask)
        d = self.rsa.decrypt(d)
        return d
    
    def send_data(self, data_bytes:bytes, mask:bytes) -> int:
        # assert len(mask) == self.nbyte
        n = len(data_bytes)
        self.socket.sendall(struct.pack('!i', n))
        if len(mask) > self.mask_len:
            mask = mask[-self.mask_len:]
        data = mask_data(data_bytes, mask)
        self.socket.sendall(data)
        return 4 + n
    

class ObliviousTransferServer():
    def __init__(self, socket:socket, nbits:int=2048) -> None:
        self.socket = socket
        self.nbits = nbits
        self.nbyte = nbits//8 + (1 if nbits % 8 > 0 else 0)
        self.rsa = RSA(nbits)
        self.mask_len = self.nbyte - 1

    def setup(self):
        data = self.socket.recv(self.nbyte)
        n = int.from_bytes(data, 'big')
        data = self.socket.recv(self.nbyte)
        e = int.from_bytes(data, 'big')
        self.rsa.setup(n, e)
        self.mask_len = self.rsa.mlength
    
    def run(self, b:int) -> tuple[bytes, int, int]:
        '''
        Oblivious transfer for receiving data with index b.
        Return the data, the number of bytes sent and received.
        '''
        assert b == 0 or b == 1
        # indicator preparation phase
        h = self.random_mask()
        k0, k1 = self.receive_pair()
        # print("Server: h =", h[:])
        c = self.rsa.encrypt(h)
        # print("Server: c =", c[:])
        if b == 0:
            c = xor_mask_char(c, k0)
        else:
            c = xor_mask_char(c, k1)
        # print("Server: c xor k_b =", c[:])
        self.socket.sendall(c)
        # data transfer phase
        m0, cnt0 = self.receive_data()
        # print("Server: m XOR h =", m0[:32])
        m1, cnt1 = self.receive_data()
        if b == 0:
            m = mask_data(m0, h)
        else:
            m = mask_data(m1, h)
        # print("Server: m =", m[:32])
        return m, self.nbyte, 2 + cnt0 + cnt1
    
    # local functions
    
    def receive_pair(self) -> tuple[int, int]:
        data = self.socket.recv(2)
        p1, p2 = struct.unpack('!BB', data)
        return p1, p2

    def receive_data(self, buf_sz:int=4096) -> tuple[bytes, int]:
        data = self.socket.recv(4)
        n = struct.unpack('!i', data)[0]
        buffer = []
        while n > 0:
            d = self.socket.recv(min(buf_sz, n))
            n -= len(d)
            buffer.append(d)
        data = b''.join(buffer)
        return data, 4 + len(data)
    
    def random_mask(self) -> bytes:
        return Crypto.Random.get_random_bytes(self.mask_len)
        # return Crypto.Random.get_random_bytes(self.nbyte)
