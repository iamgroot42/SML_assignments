import numpy as np


class HMM:
	def __init__(self, isd, opb, tpm):
		self.isd = isd
		self.opb = opb
		self.tpm = tpm
		self.os = opb.shape[1]
		self.hs = opb.shape[0]

	def getStateSequences(self, inputs):
		print("Running Viterbi Algorithm")
		V = np.zeros((inputs.shape[0], self.hs))
		# Initialize
		for i in range(self.hs):
			V[0][i] = self.opb[i][inputs[0]] * self.isd[i][0]
		# Iterate for further timesteps
		for i in range(1, inputs.shape[0]):
			for j in range(self.hs):
				V[i][j] = np.max([self.tpm[j][k] * V[i-1][k] for k in range(self.hs)]) * self.opb[j][inputs[i]]
		# Traceback to predict hidden sequence
		print("Running traceback for sequences")
		output_seq = [np.argmax(V[-1])]
		for i in range(1, inputs.shape[0]):
			output_seq.append(np.argmax([ self.tpm[k][output_seq[-1]] * V[inputs.shape[0]-(i+1)][k] for k in range(self.hs)]))
		print("Traceback generated")
		return output_seq[::-1]
