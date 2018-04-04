import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adadelta, SGD

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
keras.backend.set_session(sess)


def process_data():
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()
	# Pick only one channel
	X_train = X_train[:,:,:,0]
	X_test = X_test[:,:,:,0]
	# Flatten data
	X_train = X_train.reshape(X_train.shape[0], 1024)
	y_train = y_train.reshape(y_train.shape[0], 1)
	X_test = X_test.reshape(X_test.shape[0], 1024)
	y_test = y_test.reshape(y_test.shape[0], 1)
	# Scale down to [0,1]
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	# Convert to 1-hot
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)
	X_train = np.matrix(X_train)
	y_train = np.matrix(y_train)
	X_test = np.matrix(X_test)
	y_test = np.matrix(y_test)
	return (X_train, y_train), (X_test, y_test)


def getModel(learning_rate=1):
	model = Sequential()
	model.add(Dense(50, input_shape=(1024,)))
	model.add(Activation('relu'))
	model.add(Dense(10))
	model.add(Activation('softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,
		optimizer=Adadelta(lr=learning_rate),
		metrics=['accuracy'])
	return model


def bagging(X_train, Y_train, X_test, Y_test, n, part=0.8):
	models = [getModel() for _ in range(n)]
	for i in range(n):
		p = np.random.permutation(len(X_train))[:int(part * len(X_train))]
		Xt, Yt = X_train[p], Y_train[p]
		models[i].fit(Xt, Yt, epochs=20)
	predictions = []
	for i in range(n):
		predictions.append(models[i].predict(X_test))
	predictions = np.stack(predictions)
	predictions = np.sum(predictions, axis=0) / n
	correct = 0
	for i in range(len(Y_test)):
		if np.argmax(Y_test[i]) == np.argmax(predictions[i]):
			correct += 1
	acc = float(correct) / len(Y_test)
	return acc


if __name__ == "__main__":
	(X_train, y_train), (X_test, y_test) = process_data()
	print(bagging(X_train, y_train, X_test, y_test, 15))
