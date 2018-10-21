import numpy as np
import activation
import loss
from neural_network import *

#TODO: convert to unit tests
#TODO: add more tests

# TESTING LAYER
#- testing forward() -
#single example
layer = Layer(2, 3, activation.Linear)
X = np.array([[0,1]]).T
W = np.arange(6).reshape((3,2))
b = np.array([[1,1,1]]).T
print(X)
print(W)
print(b)
print(layer.test(layer.forward, [X], W, b))
print()

#multiple examples
layer = Layer(2, 3, activation.Linear)
X = np.eye(2)
W = np.arange(6).reshape((3,2))
b = np.array([[1,1,1]]).T
print(X)
print(W)
print(b)
print(layer.test(layer.forward, [X], W, b))


#TESTING NEURALNETWORK
#and gate
X = np.array([[0,0,1,1], \
              [0,1,0,1]])
Y = np.array([[0,0,0,1]])
nn = NeuralNetwork([2,1],loss.LogisticLoss)
nn.train(X,Y, epochs=500, learning_rate=1, output_frequency=100)
print(nn.predict(X))
print()

#or gate
X = np.array([[0,0,1,1], \
              [0,1,0,1]])
Y = np.array([[0,1,1,1]])
nn = NeuralNetwork([2,1],loss.LogisticLoss)
nn.train(X,Y, epochs=500, learning_rate=1, output_frequency=100)
print(nn.predict(X))
print()

#xor gate failure
X = np.array([[0,0,1,1], \
              [0,1,0,1]])
Y = np.array([[0,1,1,0]])
nn = NeuralNetwork([2,1],loss.LogisticLoss)
nn.train(X,Y, epochs=500, learning_rate=1, output_frequency=100)
print(nn.predict(X))
print()

#xor gate success
X = np.array([[0,0,1,1], \
              [0,1,0,1]])
Y = np.array([[0,1,1,0]])
nn = NeuralNetwork([2,2,1],loss.LogisticLoss)
nn.train(X,Y, epochs=500, learning_rate=1, output_frequency=100)
print(nn.predict(X))
