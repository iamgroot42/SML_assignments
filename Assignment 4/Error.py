import numpy as np

class Error():
	def error(self, Y, Y_cap):
		return np.sum(np.square(Y - Y_cap))/2

	def gradient(self, Y, Y_cap):
		return -(Y - Y_cap)

	def pred_error(self, Y, Y_cap):
		return 1 * (np.argmax(Y_cap[0]) != np.argmax(Y))
