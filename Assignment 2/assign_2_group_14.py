#
#  Assignment 2
#
#  Group 14:
#  <Mantra> <mantras@mun.ca>
#  <Group Member 2 name> <Group Member 1 email>
#  <Group Member 3 name> <Group Member 1 email>

####################################################################################
# Imports
####################################################################################
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#####################################################################################
#  Implementation - helper functions and question wise result generators
#####################################################################################
def load_data(file_path):
    return pd.read_csv(file_path)

def split_data(data, test_size=0.2, random_state=42):
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return rse, r2

def Q1_results():
    print('Generating results for Q1...')
    
    data = load_data('train.csv')
    
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Validation approach
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_linear_regression(X_train, y_train)
    rse_val, r2_val = evaluate_model(model, X_test, y_test)
    print(f"Validation Approach - RSE: {rse_val}, R^2: {r2_val}")
    
    # Cross-validation approach
    model_cv = LinearRegression()
    cv_scores = cross_val_score(model_cv, X, y, cv=5, scoring='neg_mean_squared_error')
    rse_cv = np.sqrt(-cv_scores.mean())
    r2_cv = cross_val_score(model_cv, X, y, cv=5, scoring='r2').mean()
    print(f"Cross-Validation Approach - RSE: {rse_cv}, R^2: {r2_cv}")

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