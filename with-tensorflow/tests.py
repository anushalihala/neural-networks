import numpy as np
import tensorflow as tf
from neural_network import *

#TODO: convert to unit tests
#TODO: add more tests (with actual datasets)

#TESTING NEURALNETWORK
#and gate
X = np.array([[0,0,1,1], \
              [0,1,0,1]])
Y = np.array([[0,0,0,1]])
loss = lambda y,z: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.transpose(z),labels=tf.transpose(y)))
nn = NeuralNetwork([2,1])
nn.train(X, Y, loss, epochs=500, learning_rate=1, output_frequency=100)
print(nn.predict(X))
print()
del nn

#or gate
X = np.array([[0,0,1,1], \
              [0,1,0,1]])
Y = np.array([[0,1,1,1]])
loss = lambda y,z: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.transpose(z),labels=tf.transpose(y)))
nn = NeuralNetwork([2,1])
nn.train(X, Y, loss, epochs=500, learning_rate=1, output_frequency=100)
print(nn.predict(X))
print()
del nn

#xor gate failure
X = np.array([[0,0,1,1], \
              [0,1,0,1]])
Y = np.array([[0,1,1,0]])
loss = lambda y,z: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.transpose(z),labels=tf.transpose(y)))
nn = NeuralNetwork([2,1])
nn.train(X, Y, loss, epochs=2000, learning_rate=0.02, output_frequency=500)
print(nn.predict(X))
print()
del nn

#xor gate success  - slow
X = np.array([[0,0,1,1], \
              [0,1,0,1]])
Y = np.array([[0,1,1,0]])
loss = lambda y,z: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.transpose(z),labels=tf.transpose(y)))
nn = NeuralNetwork([2,2,1])
nn.train(X, Y, loss, epochs=20000, learning_rate=0.02, output_frequency=500)
print(nn.predict(X))
nn.write_graph()
print()
del nn

#xor gate success  - fast
X = np.array([[0,0,1,1], \
              [0,1,0,1]])
Y = np.array([[0,1,1,0]])
loss = lambda y,z: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.transpose(z),labels=tf.transpose(y)))
nn = NeuralNetwork([2,4,2,1])
nn.train(X, Y, loss, epochs=10000, learning_rate=0.025, output_frequency=500)
print(nn.predict(X))
nn.write_graph()
print()
del nn
