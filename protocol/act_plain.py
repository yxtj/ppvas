from protocol import Protocol

class PtlActPlainPrepare(Protocol):
    def __init__(self, s):
        super().__init__(s, 'act-plain-prepare')
        
    def server_side(self):
        pass
    
    def client_side(self):
        pass
        
        
class PtlActPlainOnline(Protocol):
    def __init__(self, s):
        super().__init__(s, 'act-plain-online')
        
    def server_side(self):
        pass
    
    def client_side(self):
        pass
    