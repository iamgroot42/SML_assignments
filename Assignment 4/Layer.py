import numpy as np
import random
from helpers import sigmoid


class Layer():
	def __init__(self, weight_shape, activation, zeros=False):
		if zeros:
			self.weights = np.zeros((weight_shape[0], weight_shape[1]))
		else:
			self.weights = np.random.rand(weight_shape[0], weight_shape[1]) / 100
		self.output = np.zeros(weight_shape)
		self.gradient = None
		self.momentum = 0.0
		self.activation = activation

	def forward(self, input_data):
		self.output = np.dot(input_data, self.weights)
		return self.activation.func(self.output)
