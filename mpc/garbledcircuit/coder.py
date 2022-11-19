import torch
import io

class FixedPointCoder():
    '''
    Encode and decode a tensor to a fixed point representation (16 bits).
    '''
    
    def __init__(self, scale):
        self.scale = scale
        
    def encode(self, data: torch.Tensor):
        t = (data*self.scale).type(torch.int16)
        return t
    
    def decode(self, data: torch.Tensor, otype=torch.float32):
        t = (data.type(otype)/self.scale)
        return t
    
    def serialize(self, data: torch.Tensor) -> bytes:
        if data.dtype != torch.int16:
            data = self.encode(data)
        buffer = io.BytesIO()
        torch.save(data, buffer)
        buffer.seek(0)
        res = buffer.read()
        return res
    
    def deserialize(self, data: bytes, otype=torch.float32) -> torch.Tensor:
        buffer = io.BytesIO(data)
        buffer.seek(0)
        res = torch.load(buffer)
        if res.dtype != torch.int16:
            res = self.decode(res, otype)
        return res
    