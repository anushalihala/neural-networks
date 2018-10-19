import numpy as np
import activation

class Layer:
    def __init__(self, n_prev, n_this, activation_fn):
        self.W = np.random.randn(n_h,n_x)*0.01
        self.b = np.zeros((n_this,1))
        self.Z = np.zeros((n_this,1))

    def forward(self, ):
        pass
    
    def backward(self):
        pass


print(activation.Sigmoid.apply([9,0,-9]))
print(activation.Sigmoid.grad([9,0,-9]))
print(activation.Relu.apply([9,0,-9]))
print(activation.Relu.grad([9,0,-9]))
print(activation.Tanh.apply([9,0,-9]))
print(activation.Tanh.grad([9,0,-9]))