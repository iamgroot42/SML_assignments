import numpy as np
from viterbi import HMM
import string


def readMat(filePath):
	data = []
	with open(filePath, 'r') as f:
		for line in f:
			points = line.rstrip().split(',')
		 	data.append([float(x) for x in points])
	return np.stack(data)


def remove_adjacent(nums):
     return [a for a,b in zip(nums, nums[1:]+[not nums[-1]]) if a != b]


if __name__ == "__main__":
	import sys
	opm =  readMat('./observationProbMatrix.txt')
	tpm = readMat('./transitionProbMatrix.txt')
	isd = readMat('./initialStateDistribution.txt')
	sequence = readMat(sys.argv[1]).astype('int32')[0]
	vit = HMM(isd, opm, tpm)
	prediction = vit.getStateSequences(sequence)
	prediction = [string.ascii_lowercase[i] for i in prediction]
	with open('./out_seq.txt', 'w') as f:
		f.write(','.join(prediction))
	print("Decoded Sequence:",''.join(remove_adjacent(prediction)))
