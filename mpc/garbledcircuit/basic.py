from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


class Wire:
    def __init__(self, evalue:any, marker:int):
        self.evalue = evalue
        self.marker = marker

    
class BuildUnit:
    def __init__(self, pvalue:bool, evalue:any, marker:int):
        self.pvalue = pvalue
        self.evalue = evalue
        self.marker = marker


class Encryptor:
    def __init__(self, key:bytes, iv:bytes):
        self.key = key
        self.nbytes = len(key)
        self.iv = iv
        self.cipher = Cipher(algorithms.AES(self.key), modes.CBC(self.iv))
        self.markers = {}

    def encrypt(self, data:bytes):
        encryptor = self.cipher.encryptor()
        ct = encryptor.update(data) + encryptor.finalize()
        if ct not in self.markers:
            self.markers[ct] = len(self.markers)
        return ct, self.markers[ct]
    
    