import numpy as np


class LDA(object):
	def __init__(self, n_classes):
		self.n_classes = n_classes
		pass

	def computeSWi(self, X):
		mean = np.mean(X, axis=0)
		S = np.zeros((X.shape[1], X.shape[1]))
		for x in X:
			S += (X-mean)*(X-mean).T
		return S
	
	def computeScatters(self, X, Y):
		S_w = np.zeros((X.shape[1], X.shape[1]))
		S_b = np.zeros((X.shape[1], X.shape[1]))
		mean = np.mean(X, axis=0)
		for i in range(n_classes):
			indices = np.where(Y==i)[0]
			S_w += self.computeSWi(X[indices])
			c_mean = np.mean(X[indices], axis=0)
			S_b += len(indices) * ( (c_mean - mean) * (c_mean - mean).T)
		return S_w, S_b

	def computeDecomposition(self, X, Y):
		S_w, S_b = self.computeScatters(X, Y)
		values, vectors = np.linalg.eig(np.dot(np.linalg.inv(S_w), S_b))
		sortedIndices =  np.argsort(-values)
		self.W = []
		for i in range(len(sortedIndices)-1):
			if values[sortedIndices[i]].real > 0
				self.W.append(vectors[sortedIndices[i]].real)
		self.W = np.stack(self.W).T

	def decompose(self, X):
		return np.dot(X, self.W)

