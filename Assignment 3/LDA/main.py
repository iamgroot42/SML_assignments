import matplotlib
matplotlib.use('Agg')

from keras.datasets import cifar10
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import numpy as np
from PIL import Image
import os

import train_LDA, train_PCA


def getCifarData():
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	x_train = x_train.reshape(x_train.shape[0], 32*32*3)
	x_test = x_test.reshape(x_test.shape[0], 32*32*3)
	X = np.concatenate((x_train, x_test), axis=0)
	Y = np.concatenate((y_train, y_test), axis=0)
	return X, Y


def getFacesData(dirPath):
	X, Y = [], []
	for face in os.listdir(dirPath):
		subPath = os.path.join(dirPath, face)
		for indivFace in os.listdir(subPath):
			try:
				image = np.asarray(Image.open(os.path.join(subPath, indivFace)).resize((64, 56))) #(64, 56)
				image = image.flatten() * 1.0
				X.append(image)
				Y.append(int(face)-1)
			except:
				pass
	return np.stack(X), np.stack(Y)




def randomCVgen(X, Y, trainRatio, n=5):
	n_classes = len(np.unique(Y))
	classwise_indices = [ np.where(Y==j)[0] for j in range(n_classes)]
	for i in range(n):
		TR, TE = [], []
		for c in classwise_indices:
			indices = np.random.permutation(c)
			divide = int(len(indices) * trainRatio)
			tr, te = indices[:divide].tolist(), indices[divide:].tolist()
			TR += tr
			TE += te
		X_train, Y_train = X[TR], Y[TR]
		X_test, Y_test = X[TE], Y[TE]
		yield (X_train, Y_train), (X_test, Y_test)


if __name__ == "__main__":
	import sys
	#X, Y = getFacesData(sys.argv[1])
	X, Y = getCifarData()
	type = int(sys.argv[2])

	# Learn based on LDA representation
	plt.clf()
	plt.figure()
	scores = []
	gen = randomCVgen(X, Y, 0.7, 5)
	for i in range(5):
		(X_train, Y_train), (X_test, Y_test) = gen.next()
		if type == 1:
			lda = train_LDA.decomposeLDA(X_train, Y_train)
			X_tr_dec, X_te_dec = lda.decompose(X_train), lda.decompose(X_test)
		elif type == 2:
			pca = train_PCA.fitPCA(X_train)
	                X_tr_dec, X_te_dec = pca.transform(X_train), pca.transform(X_test)
		elif type == 3:
			pca = train_PCA.fitPCA(X_train)
			X_tr_dec, X_te_dec = pca.transform(X_train), pca.transform(X_test)
			lda = train_LDA.decomposeLDA(X_tr_dec, Y_train)
			X_tr_dec, X_te_dec = lda.decompose(X_tr_dec), lda.decompose(X_te_dec)
		else:
			lda = train_LDA.decomposeLDA(X_train, Y_train)
                        X_tr_dec, X_te_dec = lda.decompose(X_train), lda.decompose(X_test)
			pca = train_PCA.fitPCA(X_tr_dec)
			X_tr_dec, X_te_dec = pca.transform(X_tr_dec), pca.transform(X_te_dec)
		clf = svm.SVC(probability=True)
		clf.fit(X_tr_dec, Y_train)
		probs = clf.predict_proba(X_te_dec)
		predictions = []
		actual = []
		correct, total = 0, 0
		for j in range(len(X_test)):
			for k in range(len(X_test)):
				actual.append(1 * (Y_test[j] == Y_test[k]))
				predictions.append(1 - abs(np.amax(probs[j]) - np.max(probs[k])))
				total += 1
				if actual[-1] == 1 * (np.argmax(probs[j]) == np.argmax(probs[k])):
					correct += 1
		scores.append(correct / float(total))
		fpr, tpr, thresholds = roc_curve(actual, predictions)
		plt.plot(fpr, tpr, lw=2, label="ROC curve for %dth fold" % (i))
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	if type == 1:
		plt.savefig('LDA')
	elif type == 2:
		plt.savefig('PCA')
	elif type == 3:
		plt.savefig('LDA on PCA')
	else:
		plt.savefig('PCA on LDA')
	sm, std = np.mean(scores), np.std(scores)
	print("5-fold crossvalidation results: %f mean and %f s.d." % (sm, std))
