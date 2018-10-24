import numpy as np
import tensorflow as tf

#TODO: add docstrings
#TODO: support for mini-batches
#TODO: method for getting parameter values (every layer)
#TODO: better interface for loss
#TODO: better support for metrics

class Layer:
    def create(self, A_prev, n_h, activation_fn, layer_no):
        n_prev = int(A_prev.shape[0])
        self.W = tf.get_variable("W"+str(layer_no), [n_h,n_prev], initializer = tf.contrib.layers.xavier_initializer())
        self.b = tf.get_variable("b"+str(layer_no), [n_h,1], initializer = tf.zeros_initializer())
        
        Z = tf.add(tf.matmul(self.W, A_prev), self.b) 
        A = activation_fn(Z)
        
        return A
    
    def get_params(self):
        pass
        
class NeuralNetwork:
    def __init__(self, n, activation_fns=[]):
        activation_dict = {'sigmoid':tf.sigmoid,'relu':tf.nn.relu,'tanh':tf.nn.tanh,'linear':lambda x: x}
        #store params
        self.n = n
        self.activation_fns = activation_fns
        
        activation_fns = list(map(activation_dict.get, activation_fns))
        hidden_no = len(n) - 2 #omitting input and output layers
        if len(activation_fns) < hidden_no:
            diff = hidden_no - len(activation_fns)
            activation_fns += [None]*diff #pad
        #replace 'None's (i.e. incorrect or missing activations) with Relu 
        activation_fns = [activation_dict['relu'] if x is None else x for x in activation_fns]
        activation_fns.append(activation_dict['linear'])
            
        #build model
        self.X = tf.placeholder(tf.float32,shape=(n[0],None))
        self.Y = tf.placeholder(tf.float32,shape=(self.n[-1],None))
        A = self.X
        self.model=[]
        for i in range(hidden_no+1):
            A = Layer().create(A, n[i+1], activation_fns[i], i+1)
            self.model.append(A)
            
        self.Z_L = A
        self.sess = tf.Session()

    def predict(self, X_in):
        return self.sess.run( tf.sigmoid(self.Z_L), feed_dict={self.X: X_in} )
        
    def train(self, X_in, Y_in, loss_fn, optimiser_fn = None, learning_rate=0.01, epochs=1, output_frequency=5):
        loss = loss_fn(self.Y, self.Z_L)
        
        if optimiser_fn is None:
            optimiser = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
        else:
            optimiser = optimiser_fn(learning_rate=learning_rate).minimize(loss)
            
        self.sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            
            _, curr_loss = self.sess.run([optimiser, loss], feed_dict={self.X: X_in, self.Y: Y_in})
            
            if (i+1)%output_frequency==0:
                print('Cost at epoch',i+1,'=',curr_loss)
                
    def write_graph(self):
        writer = tf.summary.FileWriter('logs', self.sess.graph)
        writer.close()
                
    def __del__(self):
        self.sess.close()
        tf.reset_default_graph()