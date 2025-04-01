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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import torch
from fastai.tabular.all import *
from fastai.vision.all import *
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset




print(sys.version)
#####################################################################################
#  Implementation - helper functions and question wise result generators
#####################################################################################
def classify():
    print('Performing classification...')

def train_test_split(data, label, n_of_each_class_samples):
    scaler = MinMaxScaler(feature_range=(0, 1))  # Default range is (0,1)
    data = scaler.fit_transform(data)

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
    
    return np.array(train_X), np.array(test_X), np.array(train_Y), np.array(test_Y)

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
    train_X, test_X, train_Y, test_Y = train_test_split(data, label, n_of_each_class_samples)
    _, counts_after = np.unique(train_Y, axis=0, return_counts=True)
    print('Distribution of training labels after spliting', counts_after)
    _, counts_test = np.unique(test_Y, axis=0, return_counts=True)
    print('The test dataset percent of each classes', np.round((counts_test/counts_before)*100, 1))

def Q2_results():
    print('Generating results for Q2...')
    data = np.load('mnist_train_data.npy')
    #Find Duplicated and Missing samples
    data = data.reshape(60000,784)
    label = np.load('mnist_train_labels.npy')

    #Check if dataset is unbalanced
    _, counts_before = np.unique(label, axis=0, return_counts=True)
    
    # choose 500 samples for hyperparameter tuning, because the dataset is huge
    n_of_each_class_samples = 500
    #spliting dataset
    train_X, test_X, train_Y, test_Y = train_test_split(data, label, n_of_each_class_samples)
    
    # Define KNN
    knn = KNeighborsClassifier(metric='euclidean')
    param_grid = {'n_neighbors': np.arange(1, 100, 5)}

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(train_X, train_Y)

    # Best k value
    best_k = grid_search.best_params_['n_neighbors']
    best_score = grid_search.best_score_
    print(f"Optimal k: {best_k}")
    print(f"Best Cross-Validation Accuracy: {best_score:.4f}")

    # Find minimum number of samples for each class to statisfy second and third conditions
    n_of_each_class_samples = math.floor(min(counts_before) * 0.9)
    #spliting dataset
    train_X, test_X, train_Y, test_Y = train_test_split(data, label, n_of_each_class_samples)
    #Train KNN with best k on train data
    knn = KNeighborsClassifier(n_neighbors=best_k ,metric='euclidean')
    knn.fit(train_X, train_Y)
    print('Error of Best k on test data: ', 1 - knn.score(test_X, test_Y))

    
    # Plot accuracy vs k
    k_values = param_grid['n_neighbors']
    cv_scores = grid_search.cv_results_['mean_test_score']
    plt.plot(k_values, cv_scores, marker='o', linestyle='dashed', color='b', label="Cross-Validation Accuracy")
    plt.scatter(best_k, best_score, color='red', s=100, label=f'Best k={best_k}')

    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Accuracy")
    plt.title("KNN Hyperparameter Tuning: Accuracy vs. k")
    plt.legend()
    plt.grid(True)
    plt.show()


def Q3_results():
    print('Generating results for Q3...')
    data = np.load('mnist_train_data.npy')
    #Find Duplicated and Missing samples
    data = data.reshape(60000,784)
    label = np.load('mnist_train_labels.npy')

    #Check if dataset is unbalanced
    _, counts_before = np.unique(label, axis=0, return_counts=True)
    
    # choose 2000 samples for hyperparameter tuning, because the dataset is huge
    n_of_each_class_samples = 500
    #spliting dataset
    train_X, test_X, train_Y, test_Y = train_test_split(data, label, n_of_each_class_samples)


    param_grid = {
        'C': [0.1, 1, 10, 50, 100],
        'degree': [2, 3, 4, 5],   
        'kernel': ['poly']     
    }

    # Grid Search with Cross-Validation
    svm = SVC()
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(train_X, train_Y)

    best_C = grid_search.best_params_['C']
    best_degree = grid_search.best_params_['degree']
    print(f"Best C: {best_C}, Best degree: {best_degree}")
    best_score = grid_search.best_score_
    print(f"Best Cross-Validation Accuracy: {best_score:.4f}")

    
    # Find minimum number of samples for each class to statisfy second and third conditions
    n_of_each_class_samples = math.floor(min(counts_before) * 0.9)
    #spliting dataset
    train_X, test_X, train_Y, test_Y = train_test_split(data, label, n_of_each_class_samples)

    #Train SVM with best C and best degree on train data
    svm = SVC(kernel='poly' ,C=best_C, degree=best_degree)
    svm.fit(train_X, train_Y)
    print('Error of best C and best degree on train data: ', 1 - svm.score(test_X, test_Y))

def Q4_results():
    print('Generating results for Q4...')
    data = np.load('mnist_train_data.npy')
    #Find Duplicated and Missing samples
    data = data.reshape(60000,784)
    label = np.load('mnist_train_labels.npy')

    #Check if dataset is unbalanced
    _, counts_before = np.unique(label, axis=0, return_counts=True)
    
    # Find minimum number of samples for each class to statisfy second and third conditions
    n_of_each_class_samples = math.floor(min(counts_before) * 0.9)
    #spliting dataset
    train_X, test_X, train_Y, test_Y = train_test_split(data, label, n_of_each_class_samples)

    df_train = pd.DataFrame(train_X)
    df_train['label'] = train_Y
    df_test = pd.DataFrame(test_X)
    df_test['label'] = test_Y
    df_train['label'] = df_train['label'].astype(str)
    df_test['label'] = df_test['label'].astype(str)

    architectures = [[20, 20, 20],
                     [10, 10, 10, 10],
                     [128, 64],
                     [64, 32, 16],
                     [256, 128]]
    for layers in architectures:
        dls = TabularDataLoaders.from_df(df_train, y_names="label", bs=64, cont_names=list(df_train.columns[:-1]), shuffle_train=True)
        learn = tabular_learner(dls, layers=layers, metrics=accuracy)
        learn.lr_find()
        learn.fit_one_cycle(5)
        dl_test = learn.dls.test_dl(df_test)
        test_preds, _ = learn.get_preds(dl=dl_test)
        test_acc = accuracy(test_preds, torch.tensor(test_Y)).item()
        test_err = 1 - test_acc
        print(f"Training Data -> Architecture: {layers} - Error: {test_err:.4f}")

    # Evaluate Best amodel [256,128] on the test set
    print("Start Training Best Model")
    dls = TabularDataLoaders.from_df(df_train, y_names="label", bs=64, cont_names=list(df_train.columns[:-1]), shuffle_train=True)
    learn = tabular_learner(dls, layers=[256,128], metrics=accuracy)
    learn.lr_find()
    learn.fit_one_cycle(10)
    dl_test = learn.dls.test_dl(df_test)
    test_preds, _ = learn.get_preds(dl=dl_test)
    test_acc = accuracy(test_preds, torch.tensor(test_Y)).item()
    test_err = 1 - test_acc
    
    print(f"Test Data -> Best Architecture: [256,128] - Error: {test_err:.4f}")


    # Save the final model
    learn.export("Q4_final_model.pth")
    print("Model saved as 'final_mlp_model.pth'")


# Define a simple CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()  # Flatten layer

        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def TrainCNNModel():
    data = np.load('mnist_train_data.npy')
    #Find Duplicated and Missing samples
    data = data.reshape(60000,784)
    label = np.load('mnist_train_labels.npy')

    #Check if dataset is unbalanced
    _, counts_before = np.unique(label, axis=0, return_counts=True)
    
    # Find minimum number of samples for each class to statisfy second and third conditions
    n_of_each_class_samples = math.floor(min(counts_before) * 0.9)
    #spliting dataset
    train_X, test_X, train_Y, test_Y = train_test_split(data, label, n_of_each_class_samples)
    train_X = data.reshape(-1, 1, 28, 28)
    test_X = test_X.reshape(-1, 1, 28, 28)

    train_X = torch.tensor(train_X).float()/255.0
    train_Y = torch.tensor(label).long()

    test_X = torch.tensor(test_X).float()
    test_Y = torch.tensor(test_Y).long()

    # Create DataLoaders
    batch_size = 64
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_X, test_Y), batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    device = "cpu"
    model = CNN().to(device)
    print(f"Total parameters: {count_parameters(model)}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # model = torch.load('model.pth', weights_only=False)
    # Train the model
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Evaluate on test data
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    torch.save(model, 'Q5_model.pth')

def classifyHandwrittenDigits(Xtest, data_dir=None, model_path='Q5_model.pth'):
    model = torch.load(model_path, weights_only=False)

    Xtest = torch.tensor(Xtest, dtype=torch.float32).unsqueeze(1)/255.0

    with torch.no_grad():
        outputs = model(Xtest)
        predictions = torch.argmax(outputs, dim=1)  # Get predicted labels

    return predictions.numpy()


#########################################################################################
# Calls to generate the results
#########################################################################################
if __name__ == "__main__":
    Q1_results()
    Q2_results()
    Q3_results()
    Q4_results()
    # TrainCNNModel()

