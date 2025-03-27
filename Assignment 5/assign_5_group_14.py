#
#  Assignment 1
#
#  Group 14:
#  Ramyar Zarza rzarza@mun.ca
#  Soroush Baghernezhad sbaghernezha@mun.ca
#  Mantra mantras@mun.ca

####################################################################################
# Imports
####################################################################################
import sys
import os
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
print(sys.version)
#####################################################################################
#  Implementation - helper functions and question wise result generators
#####################################################################################
def classify():
    print('Performing classification...')

def Q1_results():
    print('Generating results for Q1...')

    data = np.load('mnist_train_data.npy')
    #Find Duplicated and Missing samples
    data = data.reshape(60000,784)
    _, counts_dup = np.unique(data, axis=0, return_counts=True)
    print('Duplicated Samples: ', np.sum(counts_dup>1))
    print('Missing Samples: ', np.sum(np.isnan(data)))
    label = np.load('mnist_train_labels.npy')

    #Check if dataset is unbalanced
    _, counts_before = np.unique(label, axis=0, return_counts=True)
    print('Distribution of labels', counts_before)
    
    # Find minimum number of samples for each class to statisfy second and third conditions
    n_of_each_class_samples = math.floor(min(counts_before) * 0.9)
    #spliting dataset
    train_X, test_X, train_Y, test_Y = [], [], [], []
    for n in range(10):
        for i in range(len(data)):
            if label[i]==n:
                if len(train_X)<n_of_each_class_samples*(n+1):
                    train_X.append(data[i])
                    train_Y.append(label[i])
                else:
                    test_X.append(data[i])
                    test_Y.append(label[i])
    _, counts_after = np.unique(train_Y, axis=0, return_counts=True)
    print('Distribution of training labels after spliting', counts_after)
    _, counts_test = np.unique(test_Y, axis=0, return_counts=True)
    print('The “test” dataset percent of each classes', np.round((counts_test/counts_before)*100, 1))

def Q2_results():
    print('Generating results for Q2...')

def Q3_results():
    print('Generating results for Q3...')

#########################################################################################
# Calls to generate the results
#########################################################################################
if __name__ == "__main__":
    Q1_results()
    Q2_results()
    Q3_results()
