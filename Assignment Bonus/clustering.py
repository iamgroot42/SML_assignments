import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering


def getDistrA():
	n = 3000
	center = [0 ,0]
	radius = 1
	angle = 2 * np.pi * np.random.randn(n, 1)
	r = radius * np.sqrt(np.random.rand(n,1))
	X1 = r * np.cos(angle)+ center[0]
	Y1 = r* np.sin(angle)+ center[1]
	circle = X1 * X1 + Y1 * Y1
	index = np.where(np.abs(circle) >= 0.6)[0]
	points = np.stack([X1[index], Y1[index]])[:,:,0]
	#plt.plot(X1[index], Y1[index], 'b.')

	n = 6000
	center = [0 ,0]
	radius = 2
	angle = 2 * np.pi * np.random.randn(n, 1)
	r = radius * np.sqrt(np.random.rand(n,1))
	X1 = r * np.cos(angle)+ center[0]
	Y1 = r* np.sin(angle)+ center[1]
	circle = X1 * X1 + Y1 * Y1
	index = np.where(np.abs(circle) >= 2.5)[0]
	points = np.concatenate((points, np.array([X1[index], Y1[index]])[:,:,0]), axis=1)
	#plt.plot(X1[index], Y1[index], 'r.')
	#plt.savefig('distributionA.png')
	return np.swapaxes(points, 0, 1)


def getDistrB():
	n = 1600
	center = [1 ,2]
	radius = 1
	angle = 2 * np.pi * np.random.randn(n, 1)
	r = radius * np.sqrt(np.random.rand(n,1))
	X1 = r * np.cos(angle)+ center[0]
	Y1 = r* np.sin(angle)+ center[1]
	points = np.stack([X1, Y1])[:,:,0]
	#plt.plot(X1, Y1, 'b.')

	n = 1600
	center = [3 ,4]
	radius = 1.5
	angle = 2 * np.pi * np.random.randn(n, 1)
	r = radius * np.sqrt(np.random.rand(n,1))
	X1 = r * np.cos(angle)+ center[0]
	Y1 = r* np.sin(angle)+ center[1]
	points = np.concatenate((points, np.array([X1, Y1])[:,:,0]), axis=1)
	#plt.plot(X1, Y1, 'r.')
	#plt.savefig('distributionB.png')
	return np.swapaxes(points, 0, 1)


if __name__ == "__main__":
	points = getDistrA()
	#labels = KMeans(n_clusters=2, random_state=0).fit_predict(points)
	labels = AgglomerativeClustering(n_clusters=2,linkage="average").fit_predict(points)
	plt.plot(points[np.where(labels==0)[0]][:,0], points[np.where(labels==0)[0]][:,1], 'r.')
	plt.plot(points[np.where(labels==1)[0]][:,0], points[np.where(labels==1)[0]][:,1], 'b.')
	#plt.savefig('Kmeans.png')
	plt.savefig('Hierarchical_Cluster.png')
