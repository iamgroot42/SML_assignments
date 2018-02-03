import argparse
import numpy as np
import _pickle as cPickle
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools

from  BayesClassifier import Bayes

parser = argparse.ArgumentParser()
parser.add_argument("testset", type=str, help="path to test.data file")
parser.add_argument("model", type=str, help="path to model file")

def load_data(filePath):
    X, Y = [], []
    with open(filePath, 'r') as f:
        for line in f:
            extracted_data = line.rstrip('\n').split(',')
            if (int(extracted_data[-1]) != 2):
                X.append(extracted_data[:-1])
                Y.append(extracted_data[-1])
    return (X, Y)

def plotROC(Y, Y_):
    fpr, tpr, threshold = metrics.roc_curve(Y, Y_)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

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
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args()
    (X, Y) = load_data(args.testset)
    with open(args.model, 'rb') as f:
        baCl = cPickle.loads(f.read())
    cm = baCl.getConfusionMatrix(X, Y)
    tn, fp, fn, tp = cm.ravel()
    print("Test Accuracy:",(tp+tn)/(tn+fp+fn+tp))
    print("TPR:", (tp)/(tp + fn))
    print("FAR:", (fp)/(fp + tn))
    (Y_pred, Y_modified) = baCl.getModifiedPredictions(X, Y)
    Y_pred = baCl.getSingletonProbs(baCl.predict(X))
    plotROC(Y_modified, Y_pred)
    plot_confusion_matrix(cm, range(2), True)
