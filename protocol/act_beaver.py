from protocol import Protocol
from mpc.beaver import SquareActivation

class PtlActBeaverPrepare(Protocol):
    def __init__(self, s):
        super().__init__(s, 'act-beaver-prepare')
        
    def server_side(self):
        pass
        
    def client_side(self):
        pass
    

class PtlActBeaverOnline(Protocol):
    def __init__(self, s):
        super().__init__(s, 'act-beaver-online')
        

    def server_side(self):
        pass
    
    def client_side(self):
        pass
    