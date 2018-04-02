from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
from Layer import Layer
from Activation import Sigmoid, SoftMax, ReLU
from Model import Model
from Error import Error

def process_data():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X_train = X_train.reshape(60000, 784)
	y_train = y_train.reshape(60000, 1)
	X_test = X_test.reshape(10000, 784)
	y_test = y_test.reshape(10000, 1)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)
	X_train = np.matrix(X_train)
	y_train = np.matrix(y_train)
	X_test = np.matrix(X_test)
	y_test = np.matrix(y_test)
	return (X_train, y_train), (X_test, y_test)


def single_layer(X_train, X_test, y_train, y_test, verbose=False):
	m = Model(Error())
	m.add_layer(Layer((784,10), SoftMax(), dropout=0.2))
	t_acc = (1-m.train(X_train, y_train, verbose)) * 100
	print "Train accuracy", t_acc, "%"
	print "Test accuracy", (1-m.test(X_test, y_test)) * 100, "%"


def getModel(i, h, o, e=None):
	m = Model(Error())
	m.add_layer(Layer((i, h), Sigmoid(), dropout=None))
	if e:
		m.add_layer(Layer((h, e), Sigmoid(), dropout=None))
		m.add_layer(Layer((e, o), SoftMax(), dropout=None))
	else:
		m.add_layer(Layer((h, o), SoftMax(), dropout=None))
	return model


t_acc = (1-m.train(X_train, y_train, verbose)) * 100
print "(", dropout_param, ",", batch_size, ",", learning_rate, ",", momentum_rate, ")"
print "Train accuracy", t_acc, "%"
print "Test accuracy", (1-m.test(X_test, y_test, 0.3)) * 100, "%"
print "-------------"


if __name__ == "__main__":
	(X_train, y_train), (X_test, y_test) = process_data()
	multi_layer(X_train, X_test, y_train, y_test, False)
