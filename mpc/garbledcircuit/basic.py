import numpy as np
from garbled_circuit import GarbledCircuit

class NBitAdder:
    def __init__(self, n):
        self.n = n
        self.circuit = GarbledCircuit(2 * n, n + 1)
        
        self.circuit.xor_gate(0, self.n, 0)
        self.circuit.and_gate(0, self.n + 1, self.n + 1)
        self.circuit.xor_gate(1, self.n + 1, self.n + 1)
        for i in range(1, self.n):
            self.circuit.xor_gate(i, self.n + i, i)
            self.circuit.and_gate(i, self.n + i + 1, self.n + i + 1)
            self.circuit.or_gate(self.n + i, self.n + i + 1, self.n + i)
            
        # Generate input and output keys
        self.input_keys, self.output_keys = self.circuit.generate_io_keys()
    
    def encrypt_inputs(self, a, b):
        # Encrypt the input values using the input keys
        return self.circuit.encrypt_inputs(self.input_keys, [a, b,  [0] * (self.n - 1)])
    
    def decrypt_outputs(self, encrypted_outputs):
        return self.circuit.decrypt_outputs(self.output_keys, encrypted_outputs)
    
    def add(self, a, b):
        return self.circuit.evaluate([a, b])
