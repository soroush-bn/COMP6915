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
def warn(*args, **kwargs):
    pass
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.catch_warnings()
warnings.warn = warn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
# from sklearn.neighbors import KNeighborsClassifier,KernelDensity
from sklearn.svm import LinearSVC,SVC
import seaborn as sns
from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_val_score, validation_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix

from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,PolynomialFeatures

#####################################################################################
#  Implementation - helper functions and question wise result generators
#####################################################################################

def data_loader(positive_dir, negative_dir):
    labels = ["ctx-lh-inferiorparietal",
"ctx-lh-inferiortemporal",
"ctx-lh-isthmuscingulate",
"ctx-lh-middletemporal",
"ctx-lh-posteriorcingulate",
"ctx-lh-precuneus",
"ctx-rh-isthmuscingulate",
"ctx-rh-posteriorcingulate",
"ctx-rh-inferiorparietal",
"ctx-rh-middletemporal",
"ctx-rh-precuneus",
"ctx-rh-inferiortemporal",
"ctx-lh-entorhinal",
"ctx-lh-supramarginal"] 
    positive_data = pd.read_csv(positive_dir, names=labels, header=None)
    positive_data['label'] = 1
    negative_data = pd.read_csv(negative_dir, names=labels, header=None)
    negative_data['label'] = 0
    
    data = pd.concat((positive_data, negative_data), axis=0, ignore_index=True)
    return data

def train_data_loader():
    return data_loader('train.fdg_pet.sDAT.csv', 'train.fdg_pet.sNC.csv')

def test_data_loader():
    return data_loader('test.fdg_pet.sDAT.csv', 'test.fdg_pet.sNC.csv')


def Q1_results():

    train = train_data_loader()
    test = test_data_loader()
    X_train = train.iloc[:,:-1]
    y_train = train.iloc[:,-1]
    X_test = test.iloc[:,:-1]
    y_test = test.iloc[:,-1]

    clf = SVC(kernel='linear', C=.5).fit(X_train, y_train)
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    param_grid = {'C': np.logspace(-3,4,50)}


    cv_accuracy_results = {}
    n_folds = [2,5,10,25,50,100]
    best_cs = []
    for cv in n_folds:
        print(f"Running GridSearchCV with {cv}-fold cross-validation...")
        
        grid_search = GridSearchCV(LinearSVC(dual=False, max_iter=5000), param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_cs.append(grid_search.best_params_['C'])        
        mean_scores = grid_search.cv_results_['mean_test_score']
        cv_accuracy_results[cv] = mean_scores

    plt.figure(figsize=(12, 6))

    for cv,c in zip(n_folds,best_cs):
        plt.semilogx(param_grid['C'], cv_accuracy_results[cv], marker='o', linestyle='-', label=f'CV={cv}')
        plt.axvline(c, linestyle="--",label=f"n_fold:{cv} -> {c:.2f}")
        
    plt.xlabel("C (log scale)")
    plt.ylabel("Cross-Validation Accuracy")
    plt.title("Effect of C on Accuracy for different number of folds")
    plt.legend()
    plt.grid(True)
    plt.show()


    final_model = LinearSVC(C=best_cs[2], dual=False, max_iter=5000)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print("performance metrics for linear kerenl SVM: ")

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall : {recall:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"Balanced Accuracy: {balanced_acc:.2f}")




def Q2_results():
   
    
    train = train_data_loader()
    test = test_data_loader()
    X_train = train.iloc[:,:-1]
    y_train = train.iloc[:,-1]
    X_test = test.iloc[:,:-1]
    y_test = test.iloc[:,-1]


    param_grid = {'C': [.1,.2,.5,.7,1,1.5,2,3,5,10,100],'degree':[1,2,3,4,5,6]}
    grid_search = GridSearchCV(SVC(kernel='poly',max_iter= 5000), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_C = grid_search.best_params_['C']
    best_degree = grid_search.best_params_['degree']
    print(f"Best C: {best_C}, Best Degree: {best_degree}")
    scores = grid_search.cv_results_['mean_test_score']
    scores_matrix = scores.reshape(len(param_grid['degree']), len(param_grid['C']))  
    plt.figure(figsize=(10, 6))
    sns.heatmap(scores_matrix, annot=True, fmt=".2f", xticklabels=np.round(param_grid['C'], 2), yticklabels=param_grid['degree'], cmap="viridis")
    plt.xlabel("C Values (log scale)")
    plt.ylabel("Polynomial Degree")
    plt.title("Grid Search Accuracy for C and Degree ")
    plt.show()


    final_model = SVC(C=best_C,kernel='poly',degree=best_degree,max_iter= 5000)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print("performance metric for poly kernel SVM: ")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall : {recall:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")



def Q3_results():

    
    train = train_data_loader()
    test = test_data_loader()
    X_train = train.iloc[:,:-1]
    y_train = train.iloc[:,-1]
    X_test = test.iloc[:,:-1]
    y_test = test.iloc[:,-1]


    param_grid = {'C': [.1,.2,.5,.7,1,1.5,2,3,5,10,100],'gamma':np.logspace(-3,0,10)}
    grid_search = GridSearchCV(SVC(kernel='rbf',max_iter= 5000), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_C = grid_search.best_params_['C']
    best_gamma = grid_search.best_params_['gamma']
    print(f"Best C: {best_C}, Best gamma: {best_gamma}")
    scores = grid_search.cv_results_['mean_test_score']
    scores_matrix = scores.reshape(len(param_grid['gamma']), len(param_grid['C'])) 
    plt.figure(figsize=(10, 6))
    sns.heatmap(scores_matrix, annot=True, fmt=".2f", xticklabels=np.round(param_grid['C'], 2), yticklabels=param_grid['gamma'], cmap="viridis")
    plt.xlabel("C Values (log scale)")
    plt.ylabel("gamma")
    plt.title("Grid Search Accuracy for C and gamma ")
    plt.show()


    final_model = SVC(C=best_C,kernel='rbf',gamma=best_gamma,max_iter= 5000)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print("performance metric for rbf kernel SVM: ")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall : {recall:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")


def noise_removal(df):
    # print(df.describe())
    dbscan = DBSCAN(eps=0.1, min_samples=4)  
    df['cluster'] = dbscan.fit_predict(df[['X1','X2']])
    df.drop(df[df['cluster'] == -1].index, inplace=True)
    return df

def Q4_results():
   
    train = train_data_loader()
    test = test_data_loader()
    X_train = train.iloc[:,:-1]
    y_train = train.iloc[:,-1]
    X_test = test.iloc[:,:-1]
    y_test = test.iloc[:,-1]

    S = MinMaxScaler()
    X_train = S.fit_transform(X_train)
    X_test= S.transform(X_test)
    param_grid = {'C': [.1,.2,.5,.7,1,1.5,2,3,5,10,100],'gamma':np.logspace(-3,0,10)}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_C = grid_search.best_params_['C']
    best_gamma = grid_search.best_params_['gamma']
    print(f"Best C: {best_C}, Best gamma: {best_gamma}")
    scores = grid_search.cv_results_['mean_test_score']
    scores_matrix = scores.reshape(len(param_grid['gamma']), len(param_grid['C'])) 

    final_model = SVC(C=best_C,kernel='rbf',gamma=best_gamma,max_iter= 5000)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print("performance metric for rbf kernel with normalization SVM: ")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall : {recall:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    
def diagnoseDAT(Xtest, data_dir):
    """Returns a vector of predictions with elements "0" for sNC and "1" for sDAT,
corresponding to each of the N_test features vectors in Xtest
Xtest N_test x 14 matrix of test feature vectors
data_dir full path to the folder containing the following files:
train.fdg_pet.sNC.csv, train.fdg_pet.sDAT.csv,
test.fdg_pet.sNC.csv, test.fdg_pet.sDAT.csv
"""

    train = data_loader(data_dir + '/train.fdg_pet.sDAT.csv', data_dir + '/train.fdg_pet.sNC.csv')
    test = data_loader(data_dir + '/test.fdg_pet.sDAT.csv', data_dir + '/test.fdg_pet.sNC.csv')
    train = pd.concat([train, test], ignore_index=True)
    X_train = train.iloc[:,:-1]
    y_train = train.iloc[:,-1]
    Xtest = np.array(Xtest).reshape(-1, 14)
    
    X_test = pd.DataFrame(Xtest, columns=X_train.columns)
    S = MinMaxScaler()
    
    X_train = S.fit_transform(X_train)

    X_test= S.transform(X_test)
    param_grid = {'C': [.1,.2,.5,.7,1,1.5,2,3,5,10,100],'gamma':np.logspace(-3,0,10)}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_C = grid_search.best_params_['C']
    best_gamma = grid_search.best_params_['gamma']

    final_model = SVC(C=best_C,kernel='rbf',gamma=best_gamma,max_iter= 5000)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    return y_pred
y_test = diagnoseDAT    
#########################################################################################
# Calls to generate the results
#########################################################################################
if __name__ == "__main__":
    # train_data_loader()
    Q1_results()
    Q2_results()
    Q3_results()
    Q4_results()
    # diagnoseDAT([1,2,3,4,5,6,7,8,9,10,1,2,3,4],".")