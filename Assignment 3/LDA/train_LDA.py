import numpy as np
from LDA import LDA


def decomposeLDA(X, Y):
	lda = LDA(len(np.unique(Y)))
	lda.computeDecomposition(X, Y)
	return lda
