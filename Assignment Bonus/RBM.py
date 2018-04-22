from sklearn.neural_network import BernoulliRBM
from keras.datasets import mnist
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = 1*(x_train >= 128)
x_test = 1*(x_test >= 128)
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

model = BernoulliRBM(256)
model.fit(x_train)
weight_matrix = np.save("RBM_weights", model.components_)
