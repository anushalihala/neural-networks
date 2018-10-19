import numpy as np

class Sigmoid:
    def apply(x):
        return 1/(1+np.exp(np.multiply(x,-1)))

    def grad(x):
        return sigmoid(x)*(1-sigmoid(x))

class Relu:
    def apply(x):
        return np.maximum(x,0)

    def grad(x):
        return np.where(np.array(x)>=0, 1, 0)

class Tanh:
    def apply(x):
        return np.tanh(x)

    def grad(x):
        return 1-tanh(x)**2
        