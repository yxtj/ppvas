from mpc.activation import Activation

'''
Beaver multiplication protocol
Prepare: 2 shares of a and b and 1 share of c=a*b
To compute x * y, send u = x - a, v = y - b to the evaluator
Since:
u*v = (x-a)*(y-b)
    = xy - x*b - y*a + a*b
    = xy - (u+a)*b - (v+b)*a + a*b
Therefore:
x*y = u*v + a*b + (u+a)*b + (v+b)*a
    = u*v + a*b + u*b + v*a
'''

class SquareActivation(Activation):
    def __init__(self, share, a, b):
        self.share = share
        self.a = a
        self.b = b
        self.ab = a*b
        self.aa = a*a
        self.bb = b*b
    
    def input_share_server(self, data):
        # u = x - a
        return data - self.a
    
    def input_share_client(self, data):
        # v = y - b
        return data - self.b
    
    def activate(self, data_s, data_c):
        # xy = u*v + a*b + u*b + v*a
        xy = data_s * data_c + self.ab + self.b*data_s + self.a*data_c
        # x*x = (u-a)*(u-a) = u*u - 2*u*a + a*a
        xx = data_s*data_s - 2*data_s*self.a + self.aa
        # y*y = (v-b)*(v-b) = v*v - 2*v*b + b*b
        yy = data_c*data_c - 2*data_c*self.b + self.bb
        # (x+y)*(x+y) = x*x + 2*x*y + y*y
        r = xx + yy + 2*xy
        return r
    
    def output_share_server(self, data):
        return data - self.share
    
    def output_share_client(self, data):
        return self.share
