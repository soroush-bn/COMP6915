
#
#  Assignment 4
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
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, balanced_accuracy_score, make_scorer
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import scipy.stats as stats
import joblib

# Define feature names present in the fdg_pet.feature.info
feature_names = [
    'ctx-lh-inferiorparietal',
    'ctx-lh-inferiortemporal',
    'ctx-lh-isthmuscingulate',
    'ctx-lh-middletemporal',
    'ctx-lh-posteriorcingulate',
    'ctx-lh-precuneus',
    'ctx-rh-isthmuscingulate',
    'ctx-rh-posteriorcingulate',
    'ctx-rh-inferiorparietal',
    'ctx-rh-middletemporal',
    'ctx-rh-precuneus',
    'ctx-rh-inferiortemporal',
    'ctx-lh-entorhinal',
    'ctx-lh-supramarginal'
]

#####################################################################################
#  Implementation - helper functions and question wise result generators
#####################################################################################
def load_data(data_dir):
    # training data
    sMCI_train = pd.read_csv(os.path.join(data_dir, 'train.fdg_pet.sMCI.csv'), header=None)
    pMCI_train = pd.read_csv(os.path.join(data_dir, 'train.fdg_pet.pMCI.csv'), header=None)
    
    # test data
    sMCI_test = pd.read_csv(os.path.join(data_dir, 'test.fdg_pet.sMCI.csv'), header=None)
    pMCI_test = pd.read_csv(os.path.join(data_dir, 'test.fdg_pet.pMCI.csv'), header=None)
    
    # Assigning the  column names to the csv files
    sMCI_train.columns = feature_names
    pMCI_train.columns = feature_names
    sMCI_test.columns = feature_names
    pMCI_test.columns = feature_names
    
    # labels - (0 for sMCI, 1 for pMCI)
    sMCI_train['label'] = 0
    pMCI_train['label'] = 1
    sMCI_test['label'] = 0
    pMCI_test['label'] = 1
    
    train_data = pd.concat([sMCI_train, pMCI_train], axis=0)
    test_data = pd.concat([sMCI_test, pMCI_test], axis=0)
    
    # Shuffling
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Spliting
    X_train = train_data.drop('label', axis=1)
    y_train = train_data['label']
    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label']
    
    return X_train, y_train, X_test, y_test

def evaluate_metrics(y_true, y_pred) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, pos_label=1)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    precision = precision_score(y_true, y_pred)
    recall = sensitivity 
    balanced_accuracy = (sensitivity + specificity) / 2
    
    return {
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'Recall': recall,
        'Balanced Accuracy': balanced_accuracy
    }

def print_metrics(metrics, title):
    print(f"\n{title} Performance Metrics:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("-" * 40)

def Q1_results():
    print('Generating results for Q1...')
    data_dir = '.'  # Current directory
    
    X_train, y_train, X_test, y_test = load_data(data_dir)
    
    #parameters
    param_grid = {
        'criterion': ['gini', 'entropy', 'log_loss']
    }
    
    # Classifier
    dt = DecisionTreeClassifier(random_state=42)
    
    # 5 fold 
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print("\nGrid Search Results for Decision Tree:")
    print("Best criterion:", grid_search.best_params_['criterion'])
    print("Best cross-validation score:", grid_search.best_score_)
    
    # retrain
    best_dt = DecisionTreeClassifier(criterion=grid_search.best_params_['criterion'], random_state=42)
    best_dt.fit(X_train, y_train)
    
    y_pred = best_dt.predict(X_test)
    metrics = evaluate_metrics(y_test, y_pred)
    
    print_metrics(metrics, "Final Decision Tree")
    
    return best_dt

def Q2_results():
    print('Generating results for Q2...')
    
    best_dt = Q1_results()
    
    plt.figure(figsize=(20, 10))
    plot_tree(best_dt, filled=True, feature_names=feature_names, class_names=['sMCI', 'pMCI'], max_depth=3)
    plt.title("Decision Tree Visualization (First 3 Levels)")
    plt.show()

    print("\nFeature Importances:")
    for feature, importance in zip(feature_names, best_dt.feature_importances_):
        print(f"{feature}: {importance:.4f}")

def Q3_results():
    print('Generating results for Q3...')
    
    data_dir = '.' 
    X_train, y_train, X_test, y_test = load_data(data_dir)
    
    param_grid = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'n_estimators': [100] 
    }
    
    # random forest classifier
    rf = RandomForestClassifier(random_state=42)
    
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print("\nGrid Search Results for Random Forest:")
    print("Best criterion:", grid_search.best_params_['criterion'])
    print("Best cross-validation score:", grid_search.best_score_)
    
    # retrain
    best_rf = RandomForestClassifier(
        criterion=grid_search.best_params_['criterion'],
        n_estimators=100,
        random_state=42
    )
    best_rf.fit(X_train, y_train)
    
    y_pred = best_rf.predict(X_test)
    metrics = evaluate_metrics(y_test, y_pred)
    
    print_metrics(metrics, "Final Random Forest")
    
    # Comparing
    dt = Q1_results()
    y_pred_dt = dt.predict(X_test)
    dt_metrics = evaluate_metrics(y_test, y_pred_dt)
    
    print("\nComparison with Decision Tree:")
    print("-" * 30)
    print("Random Forest vs Decision Tree:")
    for metric in dt_metrics:
        print(f"{metric}: {metrics[metric]:.4f} vs {dt_metrics[metric]:.4f}")
    
    return best_rf

#  Implementation - Q4 

def train_best_q4_model(X_train, y_train):
    gb = GradientBoostingClassifier(random_state=42)
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    et = ExtraTreesClassifier(random_state=42, n_jobs=-1)
    
    param_dist = {
        'gb__n_estimators': [200, 300],
        'gb__learning_rate': [0.05, 0.1],
        'rf__n_estimators': [200, 300],
        'rf__max_features': ['sqrt', None],
        'et__n_estimators': [200, 300],
        'et__max_features': ['sqrt', None]
    }
    
    ensemble = VotingClassifier([
        ('gb', gb),
        ('rf', rf),
        ('et', et)
    ], voting='soft',weights=[0.4, 0.3, 0.3])
    
    
    search = GridSearchCV(
        ensemble,
        param_dist,
        scoring='balanced_accuracy',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_

def predictMCIconverters(Xtest, data_dir):
    model_path = os.path.join(data_dir, 'q4_best_model4.pkl')
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        sMCI_train = pd.read_csv(os.path.join(data_dir, 'train.fdg_pet.sMCI.csv'), header=None)
        pMCI_train = pd.read_csv(os.path.join(data_dir, 'train.fdg_pet.pMCI.csv'), header=None)
        
        sMCI_train.columns = pMCI_train.columns = feature_names
        sMCI_train['label'] = 0
        pMCI_train['label'] = 1
        train_data = pd.concat([sMCI_train, pMCI_train], axis=0)
        X_train, y_train = train_data.drop('label', axis=1), train_data['label']
        
        model = train_best_q4_model(X_train, y_train)
        joblib.dump(model, model_path)
    
    return model.predict(Xtest)

def Q4_results():
    print('\nGenerating results for Q4...')
    data_dir = '.'
    
    X_train, y_train, X_test, y_test = load_data(data_dir)
    
    y_pred = predictMCIconverters(X_test, data_dir)
    
    metrics = evaluate_metrics(y_test, y_pred)
    print_metrics(metrics, "Q4 Best Model")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    Q1_results()
    Q2_results()
    Q3_results()
    Q4_results()

