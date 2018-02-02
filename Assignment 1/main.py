import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("file_path", type=str, help="path to tae.data file")

def load_data(filePath):
    data = []
    with open(filePath, 'r') as f:
        for line in f:
            extracted_data = line.rstrip('\n').split(',')
            if (int(extracted_data[-1]) != 2):
                data.append(line)
    return data

def dumpData(train, test):
    with open("./train.data", 'w') as f:
        for datum in train:
            f.write(datum)
    with open("./test.data", 'w') as f:
        for datum in test:
            f.write(datum)

def performSplit(data, trainRatio):
    train, test = [], []
    indices = np.random.permutation(len(data))
    trainNum = int(len(indices) * trainRatio)
    for i in range(trainNum):
        train.append(data[indices[i]])
    for i in range(len(indices) - trainNum):
        test.append(data[indices[trainNum + i]])
    return (train, test)


if __name__ == "__main__":
    args = parser.parse_args()
    data = load_data(args.file_path)
    (train, test) = performSplit(data, 0.7)
    dumpData(train, test)
