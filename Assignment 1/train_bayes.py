import argparse
import numpy as np

from  BayesClassifier import Bayes

parser = argparse.ArgumentParser()
parser.add_argument("file_path", type=str, help="path to tae.data file")

if __name__ == "__main__":

# def load_data(filePath):
#     X, Y = [], []
#     with open(filePath, 'r') as f:
#         for line in f:
#             extracted_data = line.rstrip('\n').split(',')
#             if (int(extracted_data[-1]) != 2):
#                 X.append(extracted_data[:-1])
#                 Y.append(extracted_data[-1])
#     return (X, Y)
