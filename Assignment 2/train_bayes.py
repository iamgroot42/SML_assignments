import matplotlib
matplotlib.use('Agg')

import argparse
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
import matplotlib.pyplot as plt
import itertools
from itertools import cycle

from  BayesClassifier import Bayes
from keras.datasets import mnist

parser = argparse.ArgumentParser()
parser.add_argument("alldata", type=str, help="use all data, or 3 v/s 8")


def plotROC(y_test, y_score):
    lw=2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_test.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
   	roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'wheat', 'cyan', 'lightpink', 'g', 'tomato', 'coral', 'm'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig('roc.png')


def get_data(alldata):
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()	
    X_train = X_train.reshape(60000, 784) / 64
    X_test = X_test.reshape(10000, 784) / 64
    if alldata == "yes":
        return (X_train, Y_train.tolist()), (X_test, Y_test.tolist())
    else:
        train_indices_3, train_indices_8 = np.where(Y_train==3)[0], np.where(Y_train==8)[0]
        test_indices_3, test_indices_8 = np.where(Y_test==3)[0], np.where(Y_test==8)[0]
        train_indices_3 = np.random.permutation(train_indices_3)[:0.1 * len(train_indices_3)]
        train_indices_8 = np.random.permutation(train_indices_8)[:0.9 * len(train_indices_8)]
        test_indices_3 = np.random.permutation(test_indices_3)[:0.1 * len(test_indices_3)]
        test_indices_8 = np.random.permutation(test_indices_8)[:0.9 * len(test_indices_8)]
        train_indices = np.concatenate((train_indices_3, train_indices_8), axis=0)
        test_indices = np.concatenate((test_indices_3, test_indices_8), axis=0)
        return (X_train[train_indices], Y_train[train_indices].tolist()), (X_test[test_indices], Y_test[test_indices].tolist())


def getAccuracyFromCM(cm):
    correctEntries = np.trace(cm)
    denominator = np.sum(cm)
    return float(correctEntries) / denominator


if __name__ == "__main__":
    args = parser.parse_args()
    (X_train, Y_train), (X_test, Y_test) = get_data(args.alldata)
    baCl = Bayes()
    baCl.train(X_train, Y_train)
    cm = baCl.getConfusionMatrix(X_train, Y_train)
    # Log training accuracy
    print("Training Accuracy:", getAccuracyFromCM(cm))
    # Log test accuracy
    cm = baCl.getConfusionMatrix(X_test, Y_test)
    print("Test Accuracy:", getAccuracyFromCM(cm))
    # Plot ROC curve
    (Y_pred, Y_modified) = baCl.getModifiedPredictions(X_test, Y_test)
    #Y_pred = label_binarize(Y_pred, classes=range(10))
    Y_modified = label_binarize(Y_modified, classes=range(10))
    plotROC(Y_modified, Y_pred)

