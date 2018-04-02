import numpy as np
import random
from helpers import sigmoid

class Layer():
	def __init__(self, weight_shape, activation, dropout=None, zeros=False):
		if zeros:
			self.weights = np.zeros((weight_shape[0], weight_shape[1]))
		else:
			self.weights = np.random.rand(weight_shape[0], weight_shape[1]) / 100
		self.output = np.zeros(weight_shape)
		self.dropout_mask = None
		self.gradient = None
		self.dropout = dropout
		self.momentum = 0.0
		self.activation = activation

	def forward(self, input_data):
		if self.dropout:
			self.output = input_data * np.multiply(self.weights, self.dropout_mask)
		else:
			self.output = input_data * self.weights
		return self.activation.func(self.output)

	def gen_dropout(self):
		if self.dropout:
			original_shape = self.weights.shape
			flat_weight = np.ones(original_shape)
			drop_indices = random.sample(range(flat_weight.shape[0]), int(flat_weight.shape[0] * self.dropout))
			flat_weight[drop_indices] = 0.0
			self.dropout_mask = flat_weight.reshape(original_shape)
		else:
			self.dropout_mask = np.ones(self.weights.shape)
