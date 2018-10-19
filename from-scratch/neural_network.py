import numpy as np
import activation

class Layer:
    def __init__(self, n_prev, n_h, activation_fn):
        self.n_h = n_h
        self.n_prev = n_prev
        self.W = np.random.randn(n_h,n_prev)*0.01
        self.b = np.zeros((n_h,1))
        
        self.m = None
        self.Z = None
        self.A_prev = None

    def forward(self, A_prev):
        self.m = A_prev.shape[1]      
        self.A_prev = A_prev
        
        self.Z = np.dot(W, A_prev) + b
        assert(self.Z.shape == (self.n_h , self.m))

        return activation_fn.apply(self.Z)
    
    def backward(self, dA, learning_rate):
        if self.m is None or self.Z is None or self.A_prev is None:
            print('backward() can\'t be called before forward()')
            return
    
        dZ = dA * activation_fn.grad(self.Z)
        dW = (1/m)*np.dot(dZ, self.A_prev.T)
        db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
        dA_prev = np.dot(self.W.T,dZ)

        assert(dZ.shape == dA.shape)
        assert(dW.shape == self.W.shape)
        assert(db.shape == self.b.shape)
        assert(dA_prev.shape == (self.n_prev, self.m))
        
        self.W = self.W - learning_rate*dW
        self.b = self.b - learning_rate*db

        return dA_prev
        
class NeuralNetwork:
    def __init__(self, n, activation_fns):
        activation_dict = {'sigmoid':activation.Sigmoid(),'relu':activation.Relu(),'tanh':activation.Tanh(),'linear':activation.Linear()}
        
    def predict(self, X):
        pass
        
    def train(X, y):
        pass
        
print(activation.Sigmoid.apply([9,0,-9]))
print(activation.Sigmoid.grad([9,0,-9]))
print(activation.Relu.apply([9,0,-9]))
print(activation.Relu.grad([9,0,-9]))
print(activation.Tanh.apply([9,0,-9]))
print(activation.Tanh.grad([9,0,-9]))