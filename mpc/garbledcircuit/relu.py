
class GCReLU():
    def __init__(self, nbits):
        self.nb = nbits
        
    def relu(self, data):
        if data <= 0:
            return 0
        else:
            return data
