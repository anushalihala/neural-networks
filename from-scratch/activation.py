import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(np.multiply(x,-1)))

def sigmoid_grad(x):
    return sigmoid(x)*(1-sigmoid(x))

def relu(x):
    return np.maximum(x,0)

def relu_grad(x):
    return np.where(np.array(x)>=0, 1, 0)
    
def tanh(x):
    return np.tanh(x)

def tanh_grad(x):
    return 1-tanh(x)**2