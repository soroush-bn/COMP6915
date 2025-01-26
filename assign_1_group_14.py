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

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

#####################################################################################
#  Implementation - helper functions and question wise result generators
#####################################################################################
class KNN():
    def __init__(self, distance_metric):
        if distance_metric == 'Euclidean':
            self.distance_metric = 2
        else:
            self.distance_metric = 1
        self.best_classifier = None
        self.best_classifier_accuracy = None
        self.best_classifier_k = None
        self.knn_clasifier = None

    def train_test(self, k, train_data, test_data):
        self.knn_clasifier = KNeighborsClassifier(k, p=self.distance_metric)
        self.knn_clasifier.fit(train_data[['X1', 'X2']], train_data['label'])
        train_acc = self.knn_clasifier.score(train_data[['X1', 'X2']], train_data['label'])
        test_acc = self.knn_clasifier.score(test_data[['X1', 'X2']], test_data['label'])
        
        if self.best_classifier == None:
            self.best_classifier = self.knn_clasifier
            self.best_classifier_accuracy = test_acc
            self.best_classifier_k = k
        else:
            if self.best_classifier_accuracy < test_acc:
                self.best_classifier = self.knn_clasifier
                self.best_classifier_accuracy = test_acc
                self.best_classifier_k = k

        return train_acc, test_acc
    
    def generate_grid(self, grid):
        return self.knn_clasifier.predict(grid[['X1', 'X2']])


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
        
def plot(train, test, grid, class_boundary, title):
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
    plt.scatter(grid['X1'], grid['X2'], marker='.', c=class_boundary, s=4)

    plt.title(title)      
    plt.xlim(0.6, 2.3)
    plt.xlabel('X1')
    plt.ylim(0.6, 2.3)
    plt.ylabel('X2')
    plt.show()

def Q1_results():
    global knn_Euclidean

    train = train_data_loader()
    test = test_data_loader()
    grid = grid_point_loader()

    knn_Euclidean = KNN('Euclidean')
    ks = [1,3,5,10,20,30,50,100,150,200]
    # ks = [i for i in range(1,len(train),10)]
    test_accs=  [] 
    train_accs = [] 
    for k in ks:
        train_acc, test_acc, class_boundary = *knn_Euclidean.train_test(k, train, test), knn_Euclidean.generate_grid(grid)
        test_accs.append(test_acc)
        train_accs.append(train_acc)
        title = f'k={k} - Train Error = {round(1-train_acc, 2)} - Test Error = {round(1-test_acc, 2)} - distance= Euclidean'
        plot(train, test, grid, class_boundary, title)
    print(f'test accuracies : {test_accs}')
    print(f'train accuracies : {train_accs}')

    plt.plot(ks, test_accs, 'g--', label="Test Error Rate")
    plt.plot(ks, train_accs, 'b--', label="Train Error Rate")
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.legend(loc='lower left')
    plt.show()
    

def Q2_results():
    global knn_Manhattan

    train = train_data_loader()
    test = test_data_loader()
    grid = grid_point_loader()

    knn_Manhattan = KNN('Manhattan')
    best_k = knn_Euclidean.best_classifier_k
    train_acc, test_acc, class_boundary = *knn_Manhattan.train_test(best_k, train, test), knn_Manhattan.generate_grid(grid)
    title = f'k={best_k} - Train Error = {round(1-train_acc, 2)} - Test Error = {round(1-test_acc, 2)} - distance= Manhattan'
    plot(train, test, grid, class_boundary, title)
    print(f'Euclidean -> Best Accuracy = {knn_Euclidean.best_classifier_accuracy} and Best k = {knn_Euclidean.best_classifier_k}')
    print(f'Manhattan -> Best Accuracy = {knn_Manhattan.best_classifier_accuracy} and Best k = {knn_Manhattan.best_classifier_k}')

def Q3_results():
    train = train_data_loader()
    test = test_data_loader()
    best_knn_clasifier = None
    train_error_rate = []
    test_error_rate = []
    k_neighbours = [i for i in range(100, 0, -1)]

    if knn_Euclidean.best_classifier_accuracy > knn_Manhattan.best_classifier_accuracy:
        best_knn_clasifier = KNN('Euclidean')
    else:
        best_knn_clasifier = KNN('Manhattan')

    for k in k_neighbours:
        train_acc, test_acc = best_knn_clasifier.train_test(k, train, test)
        test_error_rate.append(1-test_acc)
        train_error_rate.append(1-train_acc)
    
    plt.plot(list(map(lambda x: 1/x, k_neighbours)), test_error_rate, 'g--', label="Test Error Rate")
    plt.plot(list(map(lambda x: 1/x, k_neighbours)), train_error_rate, 'b--', label="Train Error Rate")
    plt.xscale('log')
    plt.xlabel('1/k')
    plt.ylabel('Error Rate')
    plt.legend(loc='lower left')
    plt.show()
        

#########################################################################################
# Calls to generate the results
#########################################################################################
if __name__ == "__main__":
    Q1_results()
    Q2_results()
    Q3_results()
