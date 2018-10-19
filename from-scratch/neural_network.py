import numpy as np
import activation

class Layer:
    def __init__(self, n_prev, n_this):
        s#elf.W = 
    
    def forward(self):
        pass
    
    def backward(self):
        pass


print(activation.sigmoid([9,0,-9]))
print(activation.sigmoid_grad([9,0,-9]))
print(activation.relu([9,0,-9]))
print(activation.relu_grad([9,0,-9]))
print(activation.tanh([9,0,-9]))
print(activation.tanh_grad([9,0,-9]))