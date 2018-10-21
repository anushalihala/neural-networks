import numpy as np
import activation

#TODO: ensure all matrix inputs are 2d np arrays
#TODO: add docstrings
#TODO: support for mini-batches/separate optimisation
#TODO: better interface for loss
class Layer:
    def __init__(self, n_prev, n_h, activation_fn):
        self.n_h = n_h
        self.n_prev = n_prev
        self.W = np.random.randn(n_h,n_prev)*0.01
        self.b = np.zeros((n_h,1))
        self.activation_fn = activation_fn
        
        self.m = None
        self.Z = None
        self.A_prev = None

    def forward(self, A_prev):
        self.m = A_prev.shape[1]      
        self.A_prev = A_prev
        
        self.Z = np.dot(self.W, A_prev) + self.b
        assert(self.Z.shape == (self.n_h , self.m))

        return self.activation_fn.apply(self.Z)
    
    def backward(self, dA, learning_rate):
        if self.m is None or self.Z is None or self.A_prev is None:
            print('backward() can\'t be called before forward()')
            return
    
        dZ = dA * self.activation_fn.grad(self.Z)
        dW = (1/self.m)*np.dot(dZ, self.A_prev.T)
        db = (1/self.m)*np.sum(dZ,axis=1,keepdims=True)
        dA_prev = np.dot(self.W.T,dZ)

        assert(dZ.shape == dA.shape)
        assert(dW.shape == self.W.shape)
        assert(db.shape == self.b.shape)
        assert(dA_prev.shape == (self.n_prev, self.m))
        
        self.W = self.W - learning_rate*dW
        self.b = self.b - learning_rate*db

        return dA_prev
        
    def test(self, fn, args, W=None, b=None):
        if W is not None:
            assert(W.shape == self.W.shape)
            self.W = W
        if b is not None:
            assert(b.shape == self.b.shape)
            self.b = b
            
        return fn(*args)
        
class NeuralNetwork:
    def __init__(self, n, loss, activation_fns=[]):
        #store params
        activation_dict = {'sigmoid':activation.Sigmoid,'relu':activation.Relu,'tanh':activation.Tanh,'linear':activation.Linear}
        self.n = n
        self.activation_fns = activation_fns
        self.loss = loss
        
        activation_fns = list(map(activation_dict.get, activation_fns))
        hidden_no = len(n) - 2 #omitting input and output layers
        if len(activation_fns) < hidden_no:
            diff = hidden_no - len(activation_fns)
            activation_fns += [None]*diff #pad
        #replace 'None's (i.e. incorrect or missing activations) with Relu 
        activation_fns = [activation.Relu if x is None else x for x in activation_fns]
            
        #build model
        self.model = []
        for i in range(hidden_no):
            self.model.append( Layer(n[i], n[i+1], activation_fns[i]) )
        self.model.append(Layer(n[-2], n[-1], activation.Linear))
        
    def predict(self, X):
        A = X
        for layer in self.model:
            A = layer.forward(A)
        return self.loss.apply_activation(A)
        
    def train(self,X, Y, learning_rate=0.01, epochs=1, output_frequency=5):
        for i in range(epochs):
            
            A_L = self.predict(X)
            
            if i%output_frequency==0:
                print('Cost at epoch',i+1,'=',self.loss.get_loss(Y,A_L))
            
            dA = self.loss.grad(Y,A_L)
            for layer in self.model:
                dA = layer.backward(dA, learning_rate)
        