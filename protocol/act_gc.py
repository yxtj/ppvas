from protocol import Protocol

class PtlActGcPrepare(Protocol):
    def __init__(self, s):
        super().__init__(s, 'act-gc-prepare')
        
    def server_side(self):
        pass
    
    def client_side(self):
        pass
    

class PtlActGcOnline(Protocol):
    def __init__(self, s):
        super().__init__(s, 'act-gc-online')
        
    def server_side(self):
        pass
    
    def client_side(self):
        pass
    