import numpy as np
from numpy import linalg as LA

class SVD:
	def __init__(self, eigenEnergyRatio):
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
			if eigenEnergy / totalEnergy >= self.eigenEnergy:
				break
		return chosenIndices

	def decompose(self):
		eigenValues, eigenVectors = LA.eig(data)
		eigenValues = np.diag(eigenValues)
		chosenIndices = self.eigenEnergy(eigenValues)
		self.W = eigenVectors[chosenIndices]

