#
#  Assignment 1
#
#  Group 14:
#  Ramyar Zarza rzarza@mun.ca
#  <Group Member 2 name> <Group Member 1 email>
#  <Group Member 3 name> <Group Member 1 email>

####################################################################################
# Imports
####################################################################################
import sys
import os
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt

#####################################################################################
#  Implementation - helper functions and question wise result generators
#####################################################################################
def data_loader(positive_dir, negative_dir):
    positive_data = pd.read_csv(positive_dir, names=['X1', 'X2'], header=None)
    positive_data['label'] = 1
    negative_data = pd.read_csv(negative_dir, names=['X1', 'X2'], header=None)
    negative_data['label'] = 0
    data = pd.concat((positive_data, negative_data), axis=0, ignore_index=True)
    return data

def train_data_loader():
    return data_loader('train.sDAT.csv', 'train.sNC.csv')

def test_data_loader():
    return data_loader('test.sDAT.csv', 'test.sNC.csv')

def grid_point_loader():
    return pd.read_csv('2D_grid_points.csv', names=['X1', 'X2'], header=None)

def knn_clasifier(k, train, test, grid):
    knn_clasifier = KNeighborsClassifier(k)
    knn_clasifier.fit(train[['X1', 'X2']], train['label'])
    train_acc = knn_clasifier.score(train[['X1', 'X2']], train['label'])
    test_acc = knn_clasifier.score(test[['X1', 'X2']], test['label'])
    # print(f'Train Accuracy for k={k}', train_acc)
    # print(f'Test Accuracy for k={k}', test_acc)

    return train_acc, test_acc, knn_clasifier.predict(grid[['X1', 'X2']])


def classify():
    print('Performing classification...')

def Q1_results():
    train = train_data_loader()
    test = test_data_loader()
    grid = grid_point_loader()

    for k in [1,3,5,10,20,30,50,100,150,200]:
        train_acc, test_acc, class_boundary = knn_clasifier(k, train, test, grid)
        train_label = train['label'].astype(str)
        train_label[train_label=='0'] = 'green'
        train_label[train_label=='1'] = 'blue'
        plt.scatter(train['X1'], train['X2'], marker='o', c=train_label, s=8)
        test_label = test['label'].astype(str)
        test_label[test_label=='0'] = 'green'
        test_label[test_label=='1'] = 'blue'
        plt.scatter(test['X1'], test['X2'], marker='+', c=test_label)  
        class_boundary = class_boundary.astype(str)
        class_boundary[class_boundary=='0'] = 'green'
        class_boundary[class_boundary=='1'] = 'blue'
        plt.scatter(grid['X1'], grid['X2'], marker='.', c=class_boundary, s=2)      
        plt.xlim(0.6, 2.3)
        plt.xlabel('X1')
        plt.ylim(0.6, 2.3)
        plt.ylabel('X2')
        plt.show()
        
    print('Generating results for Q1...')

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
