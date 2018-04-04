import numpy as np
import csv
import keras
import tensorflow as tf

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adadelta, SGD
from keras.callbacks import TensorBoard

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
keras.backend.set_session(sess)


def loadData(dataPath):
	X = []
	Y = []
	with open(dataPath, 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			numbers = [int(x) for x in row]
			Y.append(numbers[0])
			X.append(numbers[1:])
	return (np.stack(X).astype('float32'), np.stack(Y))


def getNetwork(input_shape, d1, d2, d3, output_shape, lr, act, d4=None):
	model = Sequential()
	model.add(Dense(d1, input_shape=input_shape))
	model.add(Activation(act))
	model.add(Dense(d2))
	model.add(Activation(act))
	model.add(Dense(d3))
	model.add(Activation(act))
	if d4:
		model.add(Dense(d4))
		model.add(Activation(act))
	model.add(Dense(output_shape))
	model.add(Activation('softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,
		optimizer=Adadelta(lr),
		metrics=['accuracy'])
	return model


def getSmallNetwork(input_shape, d1, d2, output_shape, lr):
	model = Sequential()
	model.add(Dense(d1, input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Dense(d2))
	model.add(Activation('relu'))
	model.add(Dense(output_shape))
	model.add(Activation('softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,
		optimizer=Adadelta(lr))
	return model


if __name__ == "__main__":
	# Get data
	(X_train, Y_train) = loadData('./emnist/emnist-balanced-train.csv')
	(X_test, Y_test) = loadData('./emnist/emnist-balanced-test.csv')
	X_train /= 255.
	X_test /= 255.
	Y_train = to_categorical(Y_train, 47)
	Y_test = to_categorical(Y_test, 47)
	# Part 0
	#m0 = getNetwork((784,), 256, 128, 64, 47, 0.1, 'linear')
	#m0.fit(X_train, Y_train, 32, epochs=100)
	#print(m0.evaluate(X_test, Y_test, 64)[1])
	# Part 1
	#LRs = [0.2, 0.1, 0.001]
	#accs = []
	#for lr in LRs:
	#	m1 = getNetwork((784,), 256, 128, 64, 47, lr, 'linear')
	#	m1.fit(X_train, Y_train, 32, epochs=100)
	#	accs.append(m1.evaluate(X_test, Y_test, 64)[1])
	#print(accs)
	# Part 2
	#m2 = getNetwork((784,), 512, 256, 128, 47, 0.1, 'linear')
	#m2.fit(X_train, Y_train, 32, epochs=100)
	#print(m2.evaluate(X_test, Y_test, 64)[1])
	#exit()
	# Part 3
	#m3 = getNetwork((784,), 256, 128, 64, 47, 0.1, 'linear', 128)
	#m3.fit(X_train, Y_train, 32, epochs=100)
	#print(m3.evaluate(X_test, Y_test, 64)[1])
	# Part 4
	m4 = getNetwork((784,), 256, 128, 64, 47, 0.1, 'linear')
	m4.fit(X_train, Y_train, 32, epochs=100, callbacks=[TensorBoard('./sml')])
	print(m4.evaluate(X_test, Y_test, 64)[1])
	# Part 5
	#m5 = getNetwork((784,), 256, 128, 64, 47, 0.1, 'relu')
	#m5.fit(X_train, Y_train, 32, epochs=15)
	#print(m5.evaluate(X_test, Y_test, 64)[1])
	# Part 6
	# Use best network so far, compress into 2 layer network using distillation
	#m6 = getSmallNetwork((784,), 256, 128, 47, 0.1)
	#m6.fit(X_train, Y_train, 32, epochs=20)
	#print(m6.evaluate(X_test, Y_test, 64))	
