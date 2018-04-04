import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import numpy as np
from Activation import Sigmoid, SoftMax, ReLU, Linear
from Model import Model
from Layer import Layer
from Error import Error
from keras.utils import np_utils


def loadFile(dataPath):
	X = []
	Y = []
	with open(dataPath, 'r') as f:
		for row in f:
			row = row.rstrip('\n').split(', ')
			Y.append(1*(row[-1]=='<=50K'))
			features = row[:-1]
			normal = [0, 2, 4, 10, 11, 12]
			for i in normal:
				features[i] = float(features[i])
			X.append(features)
	names = [[] for _ in range(8)]
	maps = {x:i for i, x in enumerate([1, 3, 5, 6, 7, 8, 9, 13])}
	for x in X:
		for i, j in enumerate(maps.keys()):
			names[i].append(x[j])
	names = [list(set(p)) for p in names]
	names = [{p:i for i, p in enumerate(x)} for x in names]
	for i in range(len(X)):
		for j in maps.keys():
			X[i][j] = names[maps[j]][X[i][j]]
	X = np.stack(X)
	Y = np_utils.to_categorical(np.stack(Y), 2)
	p = np.random.permutation(len(X))
	return (X[p], Y[p])


def bothSplits(X, Y):
	Y_ = np.argmax(Y, axis=1)
	X_0, Y_0 = X[np.where(Y_==0)[0]], Y[np.where(Y_==0)[0]]
	X_1, Y_1 = X[np.where(Y_==1)[0]], Y[np.where(Y_==1)[0]]
	sp = int(len(Y) * 0.5)
	X_train = np.concatenate((X_0[:sp], X_1[:sp]))
	X_test = np.concatenate((X_0[sp:], X_1[sp:]))
	Y_train = np.concatenate((Y_0[:sp], Y_1[:sp]))
	Y_test = np.concatenate((Y_0[sp:], Y_1[sp:]))
	return (X_train, Y_train), (X_test, Y_test)	


if __name__ == "__main__":
	(X, Y) = loadFile('./adult.data')
	(X_train, Y_train), (X_test, Y_test) = bothSplits(X, Y)
	m = Model(Error(), max_iters=10)
	m.add_layer(Layer((14, 3), Sigmoid()))
	m.add_layer(Layer((3, 2), SoftMax()))
	t_acc = 1-m.train(X_train, Y_train, False)
	print("Train accuracy", t_acc)
	te_acc = 1-m.test(X_test, Y_test)
	print("Test accuracy", te_acc)
	Y_ = m.predict(X_test)
	print(Y_)
