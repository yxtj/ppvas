import numpy as np
from garbled_circuit import GarbledCircuit

class NBitReLU:
    def __init__(self, n):
        self.n = n
        self.circuit = GarbledCircuit(n, n)
        self.circuit.or_gate(0, self.n, 0)
        for i in range(1, self.n):
            self.circuit.mux_gate(0, i, i + 1, i)
        self.encrypt_keys, self.decrypt_keys = self.circuit.generate_keys()
    
    def encrypt_inputs(self, x):
        return self.circuit.encrypt_inputs(self.encrypt_keys, x)
    
    def relu(self, x):
        return self.circuit.evaluate(x)
