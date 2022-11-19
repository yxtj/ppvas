from mpc.garbledcircuit.basic import Wire, BuildUnit

class OneBitAdder():
    '''
    input two bits x and y, output their sum and carry
    '''
    def __init__(self, id:int):
        self.id = id
        self.table = {}

    def setup(self, xunits:list[BuildUnit], yunits:list[BuildUnit], cunits:list[BuildUnit]):
        plain_table = {}
        for x in xunits:
            for y in  yunits:
                for c in cunits:
                    idx = (x.marker<<2) + (y.marker<<1) + c.marker
                    s = x.pvalue ^ y.pvalue ^ c.pvalue
                    c = x.pvalue & y.pvalue | x.pvalue & c.pvalue | y.pvalue & c.pvalue
                    plain_table[idx] = (s, c)
        for name, value in plain_table.items():
            self.table[name] = Wire(value, 0)
    
    def get_input_wires(self):
        pass
        
    def get_output_wires(self):
        pass

    def add(self, xwire:Wire, ywire:Wire, cwire:Wire):
        idx = (xwire.marker<<2) + (ywire.marker<<1) + cwire.marker
        return self.table[idx]
    
    
class GCAdder():
    def __init__(self, nbits):
        self.nbits = nbits
        self.circuit = []

    def setup(self, xunits:list, yunits:list):
        self.adder = OneBitAdder()
        self.adder.setup(xunits, yunits, xunits)
        c = BuildUnit(False, 0, 0)
        for i in range(self.nbits):
            adder = OneBitAdder()
            xu = xunits[i]
            yu = yunits[i]
            #cu = 
            #adder.setup(xu, yu, cu)

    def add(self, data1, data2):
        return data1 + data2
    