import numpy as np
from viterbi import HMM


def readMat(filePath):
	data = []
	with open(filePath, 'r') as f:
		for line in f:
			points = line.rstrip().split(',')
			data.append([float(x) for x in points])
	return np.stack(data)


if __name__ == "__main__":
	opm =  readMat('./observationProbMatrix.txt')
	tpm = readMat('./transitionProbMatrix.txt')
	isd = readMat('./initialStateDistribution.txt')
	sequence = readMat('./observations_art.txt').astype('int32')[0]
	vit = HMM(isd, opm, tpm)
	print vit.getStateSequences(sequence)[:10]
