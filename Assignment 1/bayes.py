import numpy as np

def load_data(filePath):
    X = [], Y = []
    with open(filePath, 'r') as f:
        for line in f:
            extracted_data = line.rstrip('\n').split(',')
            if int(extracted_data[-1] != 2):
                X.append(extracted_data[:-1])
                Y.append(extracted_data[-1])
    return (X, Y)


def performSplit(dataX, dataY, trainRatio):
    trainX, trainY = [], []
    testX, testY = [], []
    indices = np.random.permutation(len(dataX))
    trainNum = int(len(indices) * trainRatio)
    for i in range(indices[:trainNum]):
        trainX.append(dataX[indices[i]])
        trainY.append(dataY[indices[i]])
    for i in range(indices[trainNum:]):
        trainX.append(dataX[indices[i]])
        trainY.append(dataY[indices[i]])
    return (trainX, trainY), (testX, testY)
