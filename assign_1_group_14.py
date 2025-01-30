#
#  Assignment 1
#
#  Group 14:
#  Ramyar Zarza rzarza@mun.ca
#  Soroush Baghernezhad sbaghernezha@mun.ca
#  <Group Member 3 name> <Group Member 1 email>

####################################################################################
# Imports
####################################################################################
import sys
import os
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier,KernelDensity
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,PolynomialFeatures

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

    def train_test(self, k, train_data, test_data,augment=False,normalize = False,noise=False, inference=False,w='uniform'):
        self.knn_clasifier = KNeighborsClassifier(k,weights=w, p=self.distance_metric)

        # print(len(train_data))
        if augment:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            train_labels = train_data['label']
            test_labels = test_data['label']

            augmented_features = poly.fit_transform(train_data[['X1','X2']])
            feature_names = poly.get_feature_names_out(input_features=['X1', 'X2'])
            train_data = pd.DataFrame(augmented_features, columns=feature_names)
            train_data['label'] = train_labels
            augmented_features_test = poly.fit_transform(test_data[['X1','X2']])
            feature_names_test = poly.get_feature_names_out(input_features=['X1', 'X2'])
            test_data = pd.DataFrame(augmented_features_test, columns=feature_names_test)
            test_data['label'] = test_labels
            features = [[*feature_names]]


        else:
            features = [['X1','X2']]

        if noise:
            train_data = noise_removal(train_data)
        if normalize:
            scaler = MinMaxScaler()
            for f in features: 
                train_data[f] = scaler.fit_transform(train_data[f])
                test_data[f] = scaler.transform(test_data[f])
            
        self.knn_clasifier.fit(train_data[features[0]], train_data['label'])
        if inference:
            return self.knn_clasifier.predict(test_data)
        else:
            train_acc = self.knn_clasifier.score(train_data[features[0]], train_data['label'])
            test_acc = self.knn_clasifier.score(test_data[features[0]], test_data['label'])
        
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
    
    def bayes_error(self,train_data,test_data):
        kde_0 = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(train_data[train_data['label'] == 0][['X1','X2']])
        kde_1 = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(train_data[train_data['label'] == 1][['X1','X2']])

        log_p_x_given_y0 = kde_0.score_samples(test_data[['X1','X2']])
        log_p_x_given_y1 = kde_1.score_samples(test_data[['X1','X2']])

        p_x_given_y0 = np.exp(log_p_x_given_y0)
        p_x_given_y1 = np.exp(log_p_x_given_y1)

        p_y0 = np.mean(train_data['label'] == 0)
        p_y1 = np.mean(train_data['label'] == 1)

        p_y0_given_x = (p_x_given_y0 * p_y0) / (p_x_given_y0 * p_y0 + p_x_given_y1 * p_y1)
        p_y1_given_x = (p_x_given_y1 * p_y1) / (p_x_given_y0 * p_y0 + p_x_given_y1 * p_y1)

        bayes_error = np.mean(np.minimum(p_y0_given_x, p_y1_given_x))
        return bayes_error

    def generate_grid(self, grid,augment= False):
        if augment :
            return self.knn_clasifier.predict(grid[['X1', 'X2', 'X1^2', 'X1 X2', 'X2^2']])
        else:
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

def grid_point_loader(augment=False):
    df=  pd.read_csv('2D_grid_points.csv', names=['X1', 'X2'], header=None)
    if augment:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        augmented_features = poly.fit_transform(df[['X1','X2']])
        feature_names = poly.get_feature_names_out(input_features=['X1', 'X2'])
        df= pd.DataFrame(augmented_features, columns=feature_names)
    # df['X3'] = (df['X1']-df_mean_x1)**2  + ( df['X2']-df_mean_x2)**2
    return df
        
def plot(train, test, grid, class_boundary, title,x_lim= 0.6,y_lim=  2.3):
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
    plt.xlim(x_lim,y_lim)
    plt.xlabel('X1')
    plt.ylim(x_lim,y_lim)
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
    print("Estimated Bayes error is : ")
    print(knn_Euclidean.bayes_error(train,test))
    for k in ks:
        train_acc, test_acc, class_boundary = *knn_Euclidean.train_test(k, train, test), knn_Euclidean.generate_grid(grid)
        test_accs.append(test_acc)
        train_accs.append(train_acc)
        title = f'k={k} - Train Error = {round(1-train_acc, 2)} - Test Error = {round(1-test_acc, 2)} - distance= Euclidean'
        plot(train, test, grid, class_boundary, title)
    print("--"*5 +"Question 1" + "--" *5)

    print(f'test accuracies : {test_accs}')
    print(f'train accuracies : {train_accs}')
    print("--"*10)

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
    print("--"*5 +"Question 2" + "--" *5)
    print(f'Euclidean -> Best Accuracy = {knn_Euclidean.best_classifier_accuracy} and Best k = {knn_Euclidean.best_classifier_k}')
    print(f'Manhattan -> Best Accuracy = {knn_Manhattan.best_classifier_accuracy} and Best k = {knn_Manhattan.best_classifier_k}')
    print("--"*10)

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
    print("--"*5 +"Question 3" + "--" *5)
    print(f'Q3 finished')
    print("--"*10)

def noise_removal(df):
    # print(df.describe())
    dbscan = DBSCAN(eps=0.1, min_samples=4)  
    df['cluster'] = dbscan.fit_predict(df[['X1','X2']])
    df.drop(df[df['cluster'] == -1].index, inplace=True)
    return df

def Q4_results():
    global knn_Euclidean

    train = train_data_loader()
    test = test_data_loader()
    grid = grid_point_loader()
    scaler = MinMaxScaler()
    grid[['X1','X2']]   = scaler.fit_transform(grid[['X1','X2']])


    knn_Euclidean = KNN('Euclidean')
    # ks = [30]
    ks = [19]
    test_accs=  [] 
    train_accs = [] 
    for k in ks:
        train_acc, test_acc, class_boundary = *knn_Euclidean.train_test(k, train, test,normalize=True,noise=True,w= 'distance'), knn_Euclidean.generate_grid(grid)
        test_accs.append(test_acc)
        train_accs.append(train_acc)
        title = f'k={k} - Train Error = {round(1-train_acc, 2)} - Test Error = {round(1-test_acc, 2)} - distance= Euclidean'
        plot(train, test, grid, class_boundary, title,x_lim=0,y_lim=1)
    print("Question 4 accuracy:")
    print(f"test_acc:{test_accs},train_accuracy:{train_accs}")

    
def diagnoseDAT(Xtest, data_dir):
    train = data_loader(data_dir + '/train.sDAT.csv', data_dir + '/train.sNC.csv')
    test = data_loader(data_dir + '/test.sDAT.csv', data_dir + '/test.sNC.csv')
    Xtest = pd.DataFrame(Xtest, columns=['X1', 'X2'])
    knn_Euclidean = KNN('Euclidean')
    prediction = knn_Euclidean.train_test(30, train, Xtest, normalize=True, noise=True, inference=True)

    return list(prediction)

#########################################################################################
# Calls to generate the results
#########################################################################################
if __name__ == "__main__":
    Q1_results()
    Q2_results()
    Q3_results()
    Q4_results()
