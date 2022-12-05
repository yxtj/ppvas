import numpy as np
from garbled_circuit import GarbledCircuit

class NBitAdder:
    def __init__(self, n):
        self.n = n
        self.circuit = GarbledCircuit(2 * n, n + 1)
        self.encrypt_keys, self.decrypt_keys = self.circuit.generate_keys()
        
        self.circuit.xor_gate(0, self.n, 0)
        self.circuit.and_gate(0, self.n + 1, self.n + 1)
        self.circuit.xor_gate(1, self.n + 1, self.n + 1)
        for i in range(1, self.n):
            self.circuit.xor_gate(i, self.n + i, i)
            self.circuit.and_gate(i, self.n + i + 1, self.n + i + 1)
            self.circuit.or_gate(self.n + i, self.n + i + 1, self.n + i)
    
    def encrypt_inputs(self, a, b):
        return self.circuit.encrypt_inputs(self.encrypt_keys, [a, b, [0] * (self.n - 1)])

    def decrypt_outputs(self, encrypted_outputs):
        return self.circuit.decrypt_outputs(self.decrypt_keys, encrypted_outputs)

    def add(self, a, b):
        encrypted_outputs = self.circuit.evaluate([a, b])
        return self.circuit.decrypt_outputs(self.decrypt_keys, encrypted_outputs)
        
    def add_cipher(self, a, b):
        encrypted_intput = self.encrypt_inputs(a, b)
        encrypted_output = self.circuit.evaluate(encrypted_intput)
        return encrypted_output
    
    def add_plain(self, a, b):
        encrypted_intput = self.encrypt_inputs(a, b)
        encrypted_output = self.circuit.evaluate(encrypted_intput)
        return self.decrypt_outputs(encrypted_output)
