from mpc.activation import Activation

class PlainActivation(Activation):
    def __init__(self, share):
        self.share = share
    
    def output_share_server(self, data):
        return data - self.share
    
    def output_share_client(self, data):
        return self.share
