import argparse
import numpy as np
import _pickle as cPickle

from  BayesClassifier import Bayes

parser = argparse.ArgumentParser()
parser.add_argument("trainset", type=str, help="path to train.data file")
parser.add_argument("crossval", type=int, help="cross validation? (1:yes,0:no)")

def load_data(filePath):
    X, Y = [], []
    with open(filePath, 'r') as f:
        for line in f:
            extracted_data = line.rstrip('\n').split(',')
            if (int(extracted_data[-1]) != 2):
                X.append(extracted_data[:-1])
                Y.append(extracted_data[-1])
    return (X, Y)


if __name__ == "__main__":
    args = parser.parse_args()
    (X, Y) = load_data(args.trainset)
    if args.crossval != 1:
        baCl = Bayes()
        baCl.train(X, Y)
        tn, fp, fn, tp = baCl.getConfusionMatrix(X, Y).ravel()
        print("Training Accuracy:",(tp+tn)/(tn+fp+fn+tp))
        with open('model', 'wb') as f:
            # pickle.dump(baCl, f)
            f.write(cPickle.dumps(baCl, 0))
    else:
        # 5-fold cross validation
        errors = []
        for i in range(5):
            baCl = Bayes()
            X_train, Y_train = [], []
            X_val, Y_val = [], []
            for j, x in enumerate(X):
                if j % 5 != i:
                    X_train.append(x)
                    Y_train.append(Y[j])
                else:
                    X_val.append(x)
                    Y_val.append(Y[j])
            baCl.train(X_train, Y_train)
            tn, fp, fn, tp = baCl.getConfusionMatrix(X_val, Y_val).ravel()
            errors.append((tp+tn)/(tn+fp+fn+tp))
        errors = np.array(errors)
        print("Individual errors:", errors)
        print("Mean Error:", np.mean(errors))
        print("Standard Deviation (of error):", np.std(errors))
