import matplotlib
matplotlib.use('Agg')

import numpy as np
from math import erf
from scipy.integrate import quad
np.warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


def getBookData():
	samples_class1 = np.array([
	[-5.01, -8.12, -3.68]
	,[-5.43, -3.48, -3.54]
	,[1.08, -5.52, 1.66]
	,[0.86, -3.78, -4.11]
	,[-2.67, 0.63, 7.39]
	,[4.94, 3.29, 2.08]
	,[-2.51, 2.09, -2.59]
	,[-2.25, -2.13, -6.94]
	,[5.56, 2.86, -2.26]
	,[1.03, -3.33, 4.33]
	])

	samples_class2 = np.array([
	[-0.91, -0.18, -0.05]
	,[1.30, -2.06, -3.53]
	,[-7.75, -4.54, -0.95]
	,[-5.47, 0.50, 3.92]
	,[6.14, 5.72, -4.85]
	,[3.60, 1.26, 4.36]
	,[5.37, -4.63, -3.65]
	,[7.18, 1.46, -6.66]
	,[-7.39, 1.17, 6.30]
	,[-7.50, -6.32, -0.31]
	])

	samples_class3 = np.array([
	[5.35, 2.26, 8.13]
	,[5.12, 3.22, -2.66]
	,[-1.34, -5.31, -9.87]
	,[4.48, 3.42, 5.19]
	,[7.11, 2.39, 9.21]
	,[7.17, 4.33, -0.98]
	,[5.75, 3.97, 6.65]
	,[0.77, 0.27, 2.41]
	,[0.90, -0.43, -8.71]
	,[3.52, -0.36, 6.43]
	])

	data = np.array([samples_class1, samples_class2, samples_class3])
	return data


def getDeterminant(x):
	try:
		return np.linalg.det(x)
	except:
		return x


def getInverse(x):
	try:
		return np.linalg.inv(x)
	except:
		return (1/x)
	

def getDiscriminant(prior, mean, covariance):
	return lambda x: np.log(prior) - (np.log(getDeterminant(covariance)) + np.dot((x - mean).T, np.dot(getInverse(covariance), (x - mean)))) / 2


def firstQuestion(data, priors=[0.5, 0.5, 0]):
	discriminants = [ getDiscriminant(priors[i], np.mean(data[i], axis=0), np.cov(data[i].T)) for i in range(3)]
	correct, total = 0.0, 0.
	for i, perClassData in enumerate(data):
		for datum in perClassData:
			prediction = np.argmax([discriminants[j](datum) for j in range(3)])
			correct += (prediction==i)
			total += 1
	print("Emperical training error:", 1 - correct/total)
	mean1, mean2 = np.mean(data[0], axis=0), np.mean(data[1], axis=0)
	cov1, cov2 = np.cov(data[0].T), np.cov(data[1].T)
	bhatBound = np.dot((mean1-mean2).T, np.dot(getInverse(cov1+cov2),(mean1-mean2)))/8
	bhatBound += np.log(getDeterminant(cov1+cov2)/np.sqrt(getDeterminant(cov1)*getDeterminant(cov2)))/2
	bhatBound = np.sqrt(priors[0] * priors[1]) * np.exp(-bhatBound)
	print("Bhattacharyya error bound:", bhatBound)


def getRoots(a, b, c):
        d = b**2 - 4*a*c
        if d < 0:
                return None
        elif d==0:
                x = (-b+np.sqrt(b**2-4*a*c))/2*a
                return (x, x)
        else:
                x1 = (-b+np.sqrt((b**2)-(4*(a*c))))/(2*a)
                x2 = (-b-np.sqrt((b**2)-(4*(a*c))))/(2*a)
                return (x1, x2)


def thirdQuestion(means, cov, priors=[0.5, 0.5]):
	bhatBound = np.dot((means[0]-means[1]), np.dot(getInverse(cov[0]+cov[1]),(means[0]-means[1])))/8
	bhatBound + np.log(getDeterminant(cov[0]+cov[1])/np.sqrt(getDeterminant(cov[0])*getDeterminant(cov[1])))/2
	bhatBound = np.sqrt(priors[0] * priors[1]) * np.exp(-bhatBound)
	
	# Calculate k(1/2) error bound
	print("Bhattacharyya error bound:", bhatBound)
	A = (1.0/cov[0] - 1/cov[1])/2
	B = float(means[0])/cov[0] - float(means[1])/cov[1]
	C = np.log((priors[0]*np.sqrt(cov[1]))/(priors[1]*np.sqrt(cov[0]))) + ((means[1]**2/cov[1]) - (means[0]**2/cov[0]))/2
	roots = getRoots(A, B, C)

	def getWhichOne(x):
		values = [priors[i] * np.exp(-((x-means[i])**2)/(2 * cov[i])) / np.sqrt(cov[i]) for i in range(2)]
		return np.argmax(values)
	errorFunc = None
	# Get points of intersections of discriminant functions
	if not roots or (A==0 and C==0):
		# Independent of input, get class for any input point
		dominatingClass = getWhichOne(0)
		print("Error:", priors[1-dominatingClass] / np.sqrt(2))
	elif A==0:
		r = -C/B
		before=1-getWhichOne(r-1)
		after=1-getWhichOne(r+1)
		errorTerm = "erf(x-" + str(means[before]) + "/" + str(cov[before]) + ")"
		errorTerm += " - erf(x-" + str(means[after]) + "/" + str(cov[after]) + ")"
		errorTerm += " + 2erf(inf)"
		print("Error:", errorTerm)
		errorFunc = lambda x: (erf((x-means[before])/cov[before]) + 2*erf(np.inf) - erf((x-means[after])/cov[after]))/np.sqrt(8*np.pi)
	else:
		r1, r2 = np.min(roots), np.max(roots)
		before = getWhichOne(r1-1)
		middle = getWhichOne((r1+r2)/2)
		after = getWhichOne(r2+1)
		errorTerm = "erf(x-" + str(means[before]) + "/" + str(cov[before]) + ")"
		errorTerm += " - erf(x-" + str(means[after]) + "/" + str(cov[after]) + ")"
		errorTerm += " + erf(" + str((r2-means[middle])/cov[middle]) + ")"
		errorTerm += " - erf(" + str((r1-means[middle])/cov[middle]) + ")"
		print("Error:", errorTerm)
		errorFunc = lambda x: (erf((x-means[before])/cov[before]) - erf((x-means[after])/cov[after]) + erf((r2-means[middle])/cov[middle]) - erf((r1-means[middle])/cov[middle]))/np.sqrt(8*np.pi)
	
	# Use numerical methods to estimate integral (if not constant)	
	if errorFunc:
		res, _ = quad(errorFunc, -1e10, 1e10)
		res /= 2e10
		print ("Error found using numerical integration:", res)

	def getPrediction(x):
		pros = [ priors[i] * np.exp((-(x-means[i])**2)/cov[i]) / np.sqrt(cov[i]) for i in range(2)]
		return np.argmax(pros)

	def getErrorCount(numPoints):
		c1points = np.random.normal(means[0], np.sqrt(cov[0]), numPoints)	
		c2points = np.random.normal(means[1], np.sqrt(cov[1]), numPoints)
		total = 2 * numPoints
		error = 0.0
		for p in c1points:
			error += (getPrediction(p)!=0)
		for p in c2points:
			error += (getPrediction(p)!=1)
		return error / total

	pointsToTry = [10, 50, 100, 200, 500, 1000]
	errorsAtPoints = [getErrorCount(x) for x in pointsToTry]
	plt.clf()
	plt.figure(1)
	plt.plot(pointsToTry, errorsAtPoints, 'bo')
	plt.savefig(str(means)+str(cov)+str(priors)+'.png')


def secondQuestion(data, points, priors=[0.33, 0.33, 0.33]):
	distances = []
	classifications = []
	for i, perClassData in enumerate(data):
		for datum in perClassData:
			pointDistances = [np.sqrt(np.dot((datum - np.mean(data[j], axis=0)).T, np.dot(getInverse(np.cov(data[j].T)), datum-np.mean(data[j], axis=0)))) for j in range(len(data))]
			classifications.append(np.argmin(pointDistances))
			distances.append(pointDistances)
	return distances, classifications


if __name__ == "__main__":
	# First question
	print("Question 1")
	print("Using first feature:")
	firstQuestion(getBookData()[:,:,:1])
	print("Using first 2 features:")
	firstQuestion(getBookData()[:,:,:2])
	print("Using all features:")
	firstQuestion(getBookData())
	# Second question
	print("\nQuestion 2")
	points = np.array([[1,2,1], [5,3,2], [0,0,0], [1,0,0]])
	distances, classifications = secondQuestion(getBookData(), points)
	print("Using equal priors:")
	print("Distances:", distances)
	print("Classes:", classifications)
	distances, classifications = secondQuestion(getBookData(), points, priors=[0.8, 0.1, 0.1])
	print("Using unequal priors:")
        print("Distances:",distances)
	print("Classes:",classifications)
	# Third question
	print("\n Question 3")
	thirdQuestion([-0.5, 0.5], [1, 1], [0.5, 0.5])
	print("\n Question 4")
	# Fourth question
	thirdQuestion([-0.5, 0.5], [2, 2], [0.67, 0.33])
	thirdQuestion([-0.5, 0.5], [2, 2], [0.5, 0.5])
	thirdQuestion([-0.5, 0.5], [3, 1], [0.5, 0.5])
