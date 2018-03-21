from keras.datasets import cifar10
import numpy as np
from PIL import Image
import os


def getCifarData():
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	X = np.concatenate((x_train, x_test), axis=0)
	Y = np.concatenate((y_train, y_test), axis=0)
	Y = np.argmax(Y, axis=1)
	return X, Y


def getFacesData(dirPath):
	X, Y = [], []
	for face in os.listdir(dirPath):
		subPath = os.path.join(dirPath, face)
		for indivFace in os.listdir(subPath):
			try:
				image = np.asarray(Image.open(os.path.join(subPath, indivFace)))
				if image.shape[0] == 192 and image.shape[1] == 168:
					image = image.flatten() / 255.0
					X.append(image)
					Y.append(int(face))
			except:
				pass
	return np.stack(X), np.stack(Y)


def randomCVgen(X, Y, trainRatio, n=5):
	for i in range(n):
		indices = np.random.permutation(len(Y))
		divide = int(trainRatio * len(Y))
		X_train, Y_train = X[:divide], Y[:divide]
		X_test, Y_test = X[:divide], Y[:divide]
		yield (X_train, Y_train), (X_test, Y_test)


if __name__ == "__main__":
	import sys
	X, Y = getFacesData(sys.argv[1])
