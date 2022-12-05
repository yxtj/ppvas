from .activation import Activation
from .garbledcircuit.adder import GCAdder
from .garbledcircuit.relu import GCReLU

class GCActivation(Activation):
    def __init__(self, share):
        self.share = share
        
    def activate(self, data_s, data_c):
        pass
    
    def output_share_server(self, data):
        return data - self.share
    
    def output_share_client(self, data):
        return self.share