import matplotlib
matplotlib.use('Agg')
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, auc, roc_curve

import numpy as np
from Activation import Sigmoid, SoftMax, ReLU, Linear
from Model import Model
from Layer import Layer
from Error import Error
from keras.utils import np_utils


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def loadFile(dataPath):
	X = []
	Y = []
	# Read raw data
	with open(dataPath, 'r') as f:
		for row in f:
			row = row.rstrip('\n').split(', ')
			Y.append(1*(row[-1]=='<=50K'))
			features = row[:-1]
			normal = [0, 2, 4, 10, 11, 12]
			for i in normal:
				features[i] = float(features[i])
			X.append(features)
	# Map data if no mapping provided
	names = [[] for _ in range(8)]
	maps = {x:i for i, x in enumerate([1, 3, 5, 6, 7, 8, 9, 13])}
	for x in X:
		for i, j in enumerate(maps.keys()):
			names[i].append(x[j])
	names = [list(set(p)) for p in names]
	names = [{p:i for i, p in enumerate(x)} for x in names]
	for i in range(len(X)):
		for j in maps.keys():
			X[i][j] = names[maps[j]][X[i][j]]
	X = np.stack(X)
	Y = np_utils.to_categorical(np.stack(Y), 2)
	# Normalize highly varying features to a smaller range
	for i in [0, 2, 4, 10, 11, 12]:
		X[:, i] = 10 * (X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))
	#X[:, 10] = 10 * (X[:, 10] - min(X[:, 10])) / (max(X[:, 10]) - min(X[:, 10]))
	return (X, Y)


def trainTestSplit(X, Y):
	Y_zero = np.where(np.argmax(Y, axis=1)==0)[0]
	Y_one = np.where(np.argmax(Y, axis=1)==1)[0]
	split_zero = int(len(Y_zero) * 0.5)
	split_one = int(len(Y_one) * 0.5)
	trainIndices = np.concatenate((Y_zero[:split_zero], Y_one[:split_one]))
	testIndices = np.concatenate((Y_zero[split_zero:], Y_one[split_one:]))
	np.random.shuffle(trainIndices)
	np.random.shuffle(testIndices)
	return (X[trainIndices], Y[trainIndices]), (X[testIndices], Y[testIndices])


if __name__ == "__main__":
	(X, Y) = loadFile('./adult.data')
	(X_train, Y_train), (X_test, Y_test) = trainTestSplit(X, Y)
	m = Model(Error(), max_iters=100, learning_rate=1e-5)
	#m.add_layer(Layer((14, 3), ReLU()))
	m.add_layer(Layer((14, 3), Sigmoid()))
	m.add_layer(Layer((3, 2), SoftMax()))
	t_acc = 1 - m.train(X_train, Y_train, False)
	print("Train accuracy", t_acc)
	te_acc, Y_ = m.test(X_test, Y_test)
	print("Test accuracy", 1 - te_acc)
	# Plot confusion matrix
	cnf_matrix = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(Y_, axis=1))
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=['>50K', '<=50K'], normalize=True,
		title='Normalized confusion matrix')
	plt.savefig('confusionMatrix.png')
	# Plot ROC curve
	fpr, tpr, threshold = roc_curve(Y_test[:,0], Y_[:,0])
	roc_auc = auc(fpr, tpr)
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',
		lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve')
	plt.legend(loc="lower right")
	plt.savefig('roc.png')
	# Calculate EER
	fnr = 1 - tpr
	eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
	EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
	print('EER:', EER)

