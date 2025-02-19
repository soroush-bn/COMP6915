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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

#####################################################################################
#  Implementation - helper functions and question wise result generators
#####################################################################################
def load_data(file_path):
    return pd.read_csv(file_path)

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_ridge_regression(X_train, y_train, alpha):
    model = Ridge(alpha=alpha)
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
    
    # validation approach
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    linear_model = train_linear_regression(X_train, y_train)
    rse_val, r2_val = evaluate_model(linear_model, X_test, y_test)
    print(f"Simple Linear Regression - Validation RSE: {rse_val}, R^2: {r2_val}")
    
    # cross validation approach
    linear_model_cv = LinearRegression()
    cv_scores = cross_val_score(linear_model_cv, X, y, cv=5, scoring='neg_mean_squared_error')
    rse_cv = np.sqrt(-cv_scores.mean())
    r2_cv = cross_val_score(linear_model_cv, X, y, cv=5, scoring='r2').mean()
    print(f"Simple Linear Regression - Cross-Validation RSE: {rse_cv}, R^2: {r2_cv}")

def Q2_results():
    print('Generating results for Q2...')
    
    data = load_data('train.csv')
    
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    alpha_values = np.logspace(-3, 3, 50)
    param_grid = {'alpha': alpha_values}
    
    #cross-validation
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    # best alpha
    best_alpha = grid_search.best_params_['alpha']
    print(f"Best alpha: {best_alpha}")
    
    final_ridge_model = train_ridge_regression(X, y, best_alpha)
    
    test_data = load_data('test.csv')
    X_test_real = test_data.iloc[:, :-1]
    y_test_real = test_data.iloc[:, -1]
    
    #  Ridge regression model on the test set
    rse_ridge, r2_ridge = evaluate_model(final_ridge_model, X_test_real, y_test_real)
    print(f"Final Ridge Model - RSE: {rse_ridge}, R^2: {r2_ridge}")
    
    # performance vs alpha
    mse_scores = -grid_search.cv_results_['mean_test_score']
    plt.figure(figsize=(8, 5))
    plt.plot(alpha_values, np.sqrt(mse_scores), marker='o', linestyle='dashed', label='CV RSE')
    plt.axvline(best_alpha, color='r', linestyle='--', label=f'Best alpha: {best_alpha:.3f}')
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('RSE')
    plt.title('Ridge Regression: RSE vs Alpha')
    plt.legend()
    plt.show()

def Q3_results():
    print('Generating results for Q3...')

#########################################################################################
# Calls to generate the results
#########################################################################################
if __name__ == "__main__":
    Q1_results()
    Q2_results()
    Q3_results()