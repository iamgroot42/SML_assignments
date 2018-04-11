import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adadelta, SGD
from keras.models import Sequential

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
	X_train = np.matrix(X_train)
	y_train = np.matrix(y_train)
	X_test = np.matrix(X_test)
	y_test = np.matrix(y_test)
	return (X_train, y_train), (X_test, y_test)


def bagging(X_train, Y_train, X_test, Y_test, n, part=0.8):
	mlp = MLPClassifier((50,))
	bag = BaggingClassifier(mlp, n, part)
	bag.fit(X_train, Y_train)
	return bag.score(X_test, Y_test)


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


def boosting(X_train, Y_train, X_test, Y_test, n, K=10):
	sample_weights = np.ones((len(X_train),)) / len(X_train[0])
	models = [ getModel() for _ in range(n)]
	alpha =  np.ones((len(X_train),))
	y_train = np_utils.to_categorical(Y_train)
	y_test = np_utils.to_categorical(Y_test)
	for i in range(n):
		models[i].fit(X_train, y_train, sample_weight=100 * sample_weights, epochs=10)
		sw_sum = np.sum(sample_weights)
		id_term = 1 * (np.argmax(y_train, axis=1) != np.argmax(models[i].predict(X_train), axis=1))
		err = np.sum(sample_weights * id_term) / sw_sum
		alpha[i] = np.log((1 - err) / err) + np.log(K-1)
		print alpha[i], "alpha", np.log((1-err)/err), np.log(K-1)
		print err, "err"
		sample_weights = sample_weights * np.exp(alpha[i] * id_term)
		sample_weights /= np.sum(sample_weights)
		print sample_weights, "sample_weights"
	predictions = []
	for i in range(len(X_test)):
		preds = np.stack([ model.predict(X_test[i])[0] for model in models])
		for j in range(n):
			preds[j, :] *= alpha[j] 
		preds = np.sum(preds, axis=0)
		predictions.append(np.argmax(preds))
	acc = np.sum(predictions == np.argmax(y_test, axis=1))
	return float(acc) / len(Y_test)


def twoLayerNN(X_train, Y_train, X_test, Y_test, learning_rate=1):
        model = Sequential()
        model.add(Dense(400, input_shape=(1024,)))
        model.add(Activation('relu'))
	model.add(Dense(150))
	model.add(Activation('relu'))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=Adadelta(lr=learning_rate),
                metrics=['accuracy'])
        y_train = np_utils.to_categorical(Y_train)
        y_test = np_utils.to_categorical(Y_test)
	model.fit(X_train, y_train, epochs=10)
	return model.evaluate(X_test, y_test)[1]

if __name__ == "__main__":
	(X_train, y_train), (X_test, y_test) = process_data()
	#print(bagging(X_train, y_train, X_test, y_test, 5))
	#print(boosting(X_train, y_train, X_test, y_test, 5))
	print(twoLayerNN(X_train, y_train, X_test, y_test))
