import numpy as np
from numpy import linalg as LA
from tqdm import tqdm
import scipy.misc
import os


class PCA:
	def __init__(self, eigenEnergyRatio, imShape):
		self.imageShape = imShape
		self.eigenEnergyRatio = eigenEnergyRatio

	def eigenEnergy(self, eigenValues):
		sortedIndices =  np.argsort(-eigenValues)
		numEV = 0
		totalEnergy = np.sum(eigenValues)
		eigenEnergy = 0.0
		chosenIndices = []
		for i in range(len(sortedIndices)):
			eigenEnergy += eigenValues[sortedIndices[i]]
			chosenIndices.append(sortedIndices[i])
			if eigenEnergy / totalEnergy >= self.eigenEnergyRatio:
				break
		return chosenIndices

	def decompose(self, X):
		mean = np.mean(X, axis=0)
		S = np.zeros((X.shape[1], X.shape[1]))
		for i in tqdm(range(X.shape[0])):
			modifiedX = (X[i] - mean).reshape((X.shape[1], 1))
			S += np.dot(modifiedX, modifiedX.T) / X.shape[0]
		eigenValues, eigenVectors = LA.eig(S)
		eigenValues = np.real(eigenValues)
		chosenIndices = self.eigenEnergy(eigenValues)
		self.W = np.real(eigenVectors[chosenIndices])

	def transformData(self, X):
		return np.dot(X, self.W.T)


	def dumpEigenVectors(self, dir):
		for i, eigenvector in enumerate(self.W):
			image = (eigenvector.reshape(self.imageShape))
			print image
			scipy.misc.imsave(os.path.join(dir, str(i) + '.png'), image)

