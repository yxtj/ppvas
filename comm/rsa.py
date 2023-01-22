import Crypto.PublicKey.RSA
import Crypto.Util


class RSA():
    def __init__(self, nbits:int=2048, n=None, e=None) -> None:
        self.nbits = nbits
        self.nbyte = nbits//8
        self.n = None
        self.e = None
        self.d = None
    
    def setup(self, n:int=None, e:int=None):
        '''
        Setup the RSA key pair or public key.
        If n and e are given, then use them as the public key.
        Otherwise, generate a key pair (n, e) and (n, d).
        '''
        if n is not None and e is not None:
            self.n = n
            self.e = e
        else:
            key_pair = Crypto.PublicKey.RSA.generate(self.nbits)
            self.n = key_pair.n
            self.d = key_pair.d
            self.e = key_pair.e
    
    def encrypt(self, data_bytes:bytes) -> bytes:
        assert len(data_bytes) <= self.nbyte
        data_n = Crypto.Util.number.bytes_to_long(data_bytes)
        c = pow(data_n, self.e, self.n)
        mc = Crypto.Util.number.long_to_bytes(c, self.nbyte)
        return mc
    
    def decrypt(self, data_bytes:bytes) -> bytes:
        assert len(data_bytes) <= self.nbyte
        data_n = Crypto.Util.number.bytes_to_long(data_bytes)
        r = pow(data_n, self.d, self.n)
        mr = Crypto.Util.number.long_to_bytes(r, self.nbyte)
        return mr
    