#
#  Assignment 2
#
#  Group 14:
#  <Mantra> <mantras@mun.ca>
#  <Soroush Baghernezhad> <sbaghernezha@mun.ca>
#  <Ramyar Zarza> <rzarza@mun.ca>

####################################################################################
# Imports
####################################################################################
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import ElasticNetCV, LassoLarsCV, LinearRegression, Ridge, Lasso,LassoCV,RidgeCV,ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import DBSCAN
import time

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

def train_lasso_regression(X_train, y_train, alpha):
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_elastic_net_regression(X_train, y_train , alpha , l1_ratio):
    model = ElasticNet(alpha=alpha,l1_ratio=l1_ratio)
    model.fit(X_train,y_train)
    return model 

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    squared_error = np.sum((y_test - y_pred)**2)
    rse = np.sqrt(squared_error/(len(X_test) - 9))
    r2 = r2_score(y_test, y_pred)
    return rse, r2

def Q1_results():
    print('Generating results for Q1...')
    
    data = load_data('train.csv')
    
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # validation approach
    linear_model = train_linear_regression(X_train, y_train)
    # train_rse_val, train_r2_val = evaluate_model(linear_model, X_train, y_train)
    # print(f"Simple Linear Regression - Validation Approach Train RSE: {train_rse_val}, R^2: {train_r2_val}")
    test_rse_val, test_r2_val = evaluate_model(linear_model, X_test, y_test)
    print(f"Simple Linear Regression - Validation Approach Test RSE: {round(test_rse_val,2)}, R^2: {round(test_r2_val,2)}")
    
    # cross validation approach
    linear_model_cv = LinearRegression()
    n_folds = [5,10,20,50,100,len(X)]
    for n in n_folds:
        cv_scores = cross_val_score(linear_model_cv, X, y, cv=n, scoring='neg_mean_squared_error')
        
        mse_cv = -cv_scores
        rse_cv = np.sqrt(mse_cv * (len(X)) / (len(X) - 9)).mean()
        r2_cv = cross_val_score(linear_model_cv, X, y, cv=5, scoring='r2').mean()
        print(f"Simple Linear Regression - Cross-Validation for {n} folds   RSE: {round(rse_cv,2)}, R^2: {round(r2_cv,2)}")

def Q2_results():
    print('Generating results for Q2...')
    
    data = load_data('train.csv')
    
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    
    alpha_values = np.logspace(-2, 5, 50)
    param_grid = {'alpha': alpha_values}
    
    #cross-validation
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    
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
    
    data = load_data('train.csv')
    
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    
    alpha_values = np.logspace(-2, 5, 50)
    param_grid = {'alpha': alpha_values}
    
    #cross-validation
    lasso = Lasso()
    grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    
    # best alpha
    best_alpha = grid_search.best_params_['alpha']
    print(f"Best alpha: {best_alpha}")
    
    final_lasso_model = train_lasso_regression(X, y, best_alpha)
    
    test_data = load_data('test.csv')
    X_test_real = test_data.iloc[:, :-1]
    y_test_real = test_data.iloc[:, -1]
    
    #  Lasso regression model on the test set
    rse_lasso, r2_lasso = evaluate_model(final_lasso_model, X_test_real, y_test_real)
    print(f"Final Lasso Model - RSE: {rse_lasso}, R^2: {r2_lasso}")
    
    # performance vs alpha
    mse_scores = -grid_search.cv_results_['mean_test_score']
    plt.figure(figsize=(8, 5))
    plt.plot(alpha_values, np.sqrt(mse_scores), marker='o', linestyle='dashed', label='CV RSE')
    plt.axvline(best_alpha, color='r', linestyle='--', label=f'Best alpha: {best_alpha:.3f}')
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('RSE')
    plt.title('Lasso Regression: RSE vs Alpha')
    plt.legend()
    plt.show()


def remove_noise_dbscan(X, eps=50, min_samples=2):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    mask = labels != -1
    return X[mask], mask 

def remove_outliers_iqr(X):
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask = ~((X < lower_bound) | (X > upper_bound)).any(axis=1)
    return X[mask], mask

def preprocess_data(X, test=False):
    if not test : 
        X_denoised, mask = remove_outliers_iqr(X)
    else:
        X_denoised = X
        mask = None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_denoised)
    
    return X_scaled, mask  

def Q4_results():
    n_folds=20
    print('Generating results for Q4...')
    data = load_data('train.csv')
    
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    print(X.shape, y.shape)

    test_data = load_data('test.csv')
    X_test_real = test_data.iloc[:, :-1]
    y_test_real = test_data.iloc[:, -1]

    alphas = np.logspace(-2, 5, 100)  

    start_time = time.time()
    model = make_pipeline(StandardScaler() , LassoCV(alphas=alphas, cv=n_folds)).fit(X, y)
    fit_time = time.time() - start_time
    
    lasso = model[-1]
    n, p = X.shape
    rse_path = np.sqrt((lasso.mse_path_ * n) / (n - p - 1))
    plt.subplot(1, 2, 1)
    plt.semilogx(lasso.alphas_, rse_path, linestyle=":")
    plt.plot(lasso.alphas_, rse_path.mean(axis=-1), color="black", label="Average across the folds", linewidth=2)
    plt.axvline(lasso.alpha_, linestyle="--", color="black", label="alpha: CV estimate")
    plt.xlabel(r"$\alpha$")
    plt.ylabel("(RSE)")
    plt.legend()
    _ = plt.title(f"Lasso via coordinate descent")
    final_lasso_model = train_lasso_regression(X, y, lasso.alpha_)
    rse_lasso, r2_lasso = evaluate_model(final_lasso_model, X_test_real, y_test_real)
    print("best lasso alpha: "  + str(lasso.alpha_))
    print(f"Final lasso Model - RSE: {rse_lasso}, R^2: {r2_lasso}")
    
    # RidgeCV
    start_time = time.time()
    model = make_pipeline(StandardScaler() ,RidgeCV(alphas=alphas, cv=n_folds )).fit(X, y)
    fit_time = time.time() - start_time
    ridge = model[-1]
    print("best ridge alpha" + str(ridge.alpha_))
        
    final_ridge_model = train_ridge_regression(X, y, ridge.alpha_)
    
    rse_ridge, r2_ridge = evaluate_model(final_ridge_model, X_test_real, y_test_real)
    print(f"Final Ridge Model - RSE: {rse_ridge}, R^2: {r2_ridge}")
    

    # ElasticNetCV(
    start_time = time.time()
    model = make_pipeline(StandardScaler() , ElasticNetCV(alphas=alphas, cv=n_folds)).fit(X, y)
    fit_time = time.time() - start_time
    
    elastic_net = model[-1]
    print("best elasticNET alpha : " + str(elastic_net.alpha_))
    print("best elasticNET l1 ratio : " + str(elastic_net.l1_ratio_))
    final_elastic_net_model = train_elastic_net_regression(X_train= X, y_train=y, alpha=elastic_net.alpha_,l1_ratio=elastic_net.l1_ratio_)
    
    rse_elastic_net, r2_elastic_net = evaluate_model(final_elastic_net_model, X_test_real, y_test_real)
    print(f"Final elastic_net Model - RSE: {rse_elastic_net}, R^2: {r2_elastic_net}")
    n, p = X.shape
    rse_path = np.sqrt((elastic_net.mse_path_ * n) / (n - p - 1))
    plt.subplot(1, 2, 2)
    plt.semilogx(elastic_net.alphas_, rse_path, linestyle=":")
    plt.plot(elastic_net.alphas_, rse_path.mean(axis=-1), color="black", label="Average across the folds", linewidth=2)
    plt.axvline(elastic_net.alpha_, linestyle="--", color="black", label="alpha: CV estimate")
    plt.xlabel(r"$\alpha$")
    plt.ylabel("(RSE)")
    plt.legend()
    _ = plt.title(f"ElasticNet")
    plt.show()

def ytest(x_test, data_dir):
    n_folds=20
    print('Generating results for Q4...')
    data = load_data(data_dir)
    
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    x_test = pd.DataFrame(x_test, columns=data.columns[:8])
    alphas = np.logspace(-2, 5, 100)  
    # RidgeCV
    start_time = time.time()
    model = make_pipeline(StandardScaler() ,RidgeCV(alphas=alphas, cv=n_folds )).fit(X, y)
    fit_time = time.time() - start_time
    ridge = model[-1]
 
    final_ridge_model = train_ridge_regression(X, y, ridge.alpha_)
    return final_ridge_model.predict(x_test)


#########################################################################################
# Calls to generate the results
#########################################################################################
if __name__ == "__main__":
    Q1_results()
    Q2_results()
    Q3_results()
    Q4_results()