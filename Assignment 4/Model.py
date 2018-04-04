import numpy as np
from Layer import Layer


class Model():
	def __init__(self, error, learning_rate = 1e-3, gamma = 0.5, batch_size = 128, max_iters=50):
		self.layers = []
		self.error = error
		self.max_iters = max_iters
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.batch_size = batch_size

	def add_layer(self, layer):
		self.layers.append(layer)

	def forward_pass(self):
		self.layers[0].forward(self.X)
		for i in range(1,len(self.layers)):
			self.layers[i].forward(self.layers[i-1].output)
		self.Y_cap = self.layers[-1].activation.func(self.layers[-1].output)

	def backward_pass(self):
		self.layers[-1].gradient  = self.error.gradient(self.Y, self.Y_cap).T
		for i in range(len(self.layers)-2,-1,-1):
			activation_gradient = self.layers[i].activation.gradient(self.layers[i].output).T
			second_term = np.dot(self.layers[i+1].weights, self.layers[i+1].gradient)
			self.layers[i].gradient = np.multiply(activation_gradient, second_term)

	def update_weights(self):
		gradient = self.layers[0].gradient * self.X
		self.layers[0].momentum *= self.gamma
		self.layers[0].momentum += self.learning_rate * (gradient.T)
		self.layers[0].weights -= self.layers[0].momentum
		for i in range(1,len(self.layers)):
			gradient = self.layers[i].gradient * self.layers[i-1].activation.func(self.layers[i-1].output)
			self.layers[i].momentum *= self.gamma
			self.layers[i].momentum += self.learning_rate * (gradient.T)
			self.layers[i].weights -= self.layers[i].momentum

	def one_iter(self, test=False):
		self.forward_pass()
		if not test:
			self.backward_pass()
			self.update_weights()
		return self.error.pred_error(self.Y, self.Y_cap)

	def train(self, X_train, Y_train, verbose=False):
		training_accuracy = 0.0
		num_batches = 0
		for i in range(0,len(X_train)-self.batch_size,self.batch_size):
			total_error = 0.0
			X_batch = X_train[i: i+self.batch_size]
			Y_batch = Y_train[i: i+self.batch_size]
			prev_error = 1.0
			num_iters = 0
			for _ in range(self.max_iters):
				batch_pass_error = 0.0
				for j in range(self.batch_size):
					self.X = X_batch[j:j+1]
					self.Y = Y_batch[j]
					error = self.one_iter()
					batch_pass_error += error
				batch_pass_error /= float(self.batch_size)
				num_iters += 1
				if prev_error - batch_pass_error <= 0.001:
					break
				prev_error = batch_pass_error
			total_error += batch_pass_error
			total_error /= float(num_iters)
			training_accuracy += total_error
			num_batches += 1
			if verbose:
				print "Accuracy for this batch", (1-total_error) * 100, "%"
		return training_accuracy / float(num_batches)

	def test(self, X_test, y_test):
		total_acc = 0.0
		for i in range(len(X_test)):
			self.X = X_test[i]
			self.Y = y_test[i]
			total_acc += self.one_iter(True)
		total_acc /= float(len(X_test))
		return total_acc

	def predict(self, X_test):
		predictions = []
		for i in range(len(X_test)):
			self.X = X_test[i]
			self.layers[0].forward(self.X)
	                for i in range(1,len(self.layers)):
        	                self.layers[i].forward(self.layers[i-1].output)
                	self.Y_cap = self.layers[-1].activation.func(self.layers[-1].output)			
			predictions.append(self.Y_cap)
		return np.stack(predictions)
