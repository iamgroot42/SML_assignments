import numpy as np
from scipy.misc import imresize
from PIL import Image
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from PCA import PCA


def readAllImages(directory, newShape):
	X = []
	Y = []
	for label in os.listdir(directory):
		subDir = os.path.join(directory, label)
		for imagePath in os.listdir(subDir):
			try:
				image = np.asarray(Image.open(os.path.join(subDir, imagePath)))
				image = imresize(image, newShape).flatten() / 255.0
				X.append(image)
				Y.append(label)
			except:
				pass
	X_matrix = np.zeros((len(X), len(X[0])))
	for i, row in enumerate(X):
		for j, entry in enumerate(row):
			X_matrix[i][j] = entry
	return (X_matrix, np.array(Y))


if __name__ == "__main__":
	import sys
	newShape = (112, 98)
	(X, Y) = readAllImages(sys.argv[1], newShape)
	# Train classifier on original data
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)
	clf.fit(X_train, Y_train)
	print('Accuracy for classifier using original data', clf.score(X_test, Y_test))
 	# 90% eigenenergy
	pca = PCA(0.9, newShape)
	pca.decompose(X)
	pca.dumpEigenVectors(sys.argv[2])
	# 95% eigenenergy
	pca = PCA(0.95, newShape)
	pca.decompose(X)
	pca.dumpEigenVectors(sys.argv[3])
	# 99% eigenenergy
	pca  = PCA(0.99, newShape)
	pca.decompose(X)
	pca.dumpEigenVectors(sys.argv[4]) 
	# Train classifier on reduces dimensionality data
	X_new = pca.transformData(X)
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
	X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y, test_size=0.5)
	clf.fit(X_train, Y_train)
	print('Accuracy for classifier using transformed data', clf.score(X_test, Y_test))
