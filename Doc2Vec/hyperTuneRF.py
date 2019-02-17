# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 09:26:22 2019

@author: kishite
"""
from numba import jit
from numba import vectorize
from numba import cuda

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from createData import Data

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

from pprint import pprint

import numpy as np

RANDOM_SEED = 7

class tuneRF():
    
    def __init__(self):
    
        """
        Initalize 
        """
        self.data = Data()
        
        #collections.Counter(ranks)    

    """
    Read file
    """
    def readfile(self, fname):
        
        df = self.data.readCorp(fname)
        print(df.shape)
        
        # Split data into training and target variables
        #Section to data and label
        data_X = df.iloc[:, 2:-1].values
        data_Y = df.iloc[:, 1].values
        return(data_X, data_Y)  
        
    """
    Split dataset into train and test
    """
    @jit
    def split(self):
        
        df = self.data.readCorp(r'DataFeatureSet\export_feature_all.csv')
        print(df.shape)
        
        # Split data into training and target variables
        #Section to data and label
        data_X = df.iloc[:, 2:-1].values
        data_Y = df.iloc[:, 1].values

        # split the dataset into training and validation datasets 
        #Split dataset into training and testing set
        X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size = 0.2, random_state=RANDOM_SEED)
        #train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df)

        # label encode the target variable 
#        encoder = preprocessing.LabelEncoder()
#        y_train = encoder.fit_transform(y_train)
#        y_test = encoder.fit_transform(y_test)
        
        return ( X_train, X_test, y_train, y_test)
   
    """
    Default Random Forest for performance
    """
    def defaultRF(self, x_train, y_train, x_test, y_test):
        rf = RandomForestRegressor()
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test)
        return(y_pred)
        
    """
    Performace using AUC
    """
    def aucRF(self, y_test, y_pred):
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        
        return(roc_auc)
        
    """
    Number of Estimators
    """
#    def numEstimators(self, X_train, X_test, y_train, y_test):
#        n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200, 1000]
#
#        train_results = []
#        test_results = []
#        for estimator in n_estimators:
#           rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
#           rf.fit(X_train, y_train)
#        
#           train_pred = rf.predict(X_train)
#        
#           false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
#           roc_auc = auc(false_positive_rate, true_positive_rate)
#           train_results.append(roc_auc)
#        
#           y_pred = rf.predict(X_test)
#        
#           false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
#           roc_auc = auc(false_positive_rate, true_positive_rate)
#           test_results.append(roc_auc)
#        
#        line1, = plt.plot(n_estimators, train_results, 'b', label=”Train AUC”)
#        line2, = plt.plot(n_estimators, test_results, 'r', label=”Test AUC”)
#        
#        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
#        
#        plt.ylabel(‘AUC score’)
#        plt.xlabel(‘n_estimators’)
#        plt.show()
        #return(roc_auc)
        
    """
    Max Depth
    """
#    def maxDepth(self, X_train, X_test, y_train, y_test):
#        max_depths = np.linspace(1, 32, 32, endpoint=True)
#
#        train_results = []
#        test_results = []
#        for max_depth in max_depths:
#           rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
#           rf.fit(x_train, y_train)
#        
#           train_pred = rf.predict(x_train)
#        
#           false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
#           roc_auc = auc(false_positive_rate, true_positive_rate)
#           train_results.append(roc_auc)
#        
#           y_pred = rf.predict(x_test)
#        
#           false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
#           roc_auc = auc(false_positive_rate, true_positive_rate)
#           test_results.append(roc_auc)
#             
#        line1, = plt.plot(max_depths, train_results, 'b', label=”Train AUC”)
#        line2, = plt.plot(max_depths, test_results, 'r', label=”Test AUC”)
#        
#        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
#        
#        plt.ylabel(‘AUC score’)
#        plt.xlabel(‘Tree depth’)
#        plt.show()
    
    """
    Create Random Grid for hyperparameter tuning
    """
    @jit
    def randomGrid(self):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        
        pprint(random_grid)
        return(random_grid)
        
        
    """
    Train Random Grid for hyperparameter tuning
    """
    @jit
    @vectorize(['float32(float32, float32)'], target='cuda')
    def trainGrid(self,X_train, y_train, random_grid):
        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestRegressor()
        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
        
        # Fit the random search model
        rf_random.fit(X_train, y_train)
        rf_random.best_params_
        return(rf_random)
    
    """
    Evaluate 
    """ 
    @jit
    @cuda.jit(device=True)
    def evaluate(self, model, test_features, test_labels):
        predictions = model.predict(test_features)
        errors = abs(predictions - test_labels)
        mape = 100 * np.mean(errors / test_labels)
        accuracy = 100 - mape
        print('Model Performance')
        print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
        print('Accuracy = {:0.2f}%.'.format(accuracy))
        
        return accuracy
        
    """
    Evaluate Random search by comparing with base model
    """
    @jit
    @vectorize(['float32(float32, float32, float32, float32)'], target='cuda')
    def evalRand(self,X_train, y_train, X_test, y_test, rf_random):
        base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
        base_model.fit(X_train, y_train)
        base_accuracy = self.evaluate(base_model, X_test, y_test)
        
        best_random = rf_random.best_estimator_
        random_accuracy = self.evaluate(best_random, X_test, y_test)
        
        print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
        
        
    
    
if __name__ == "__main__":
    tune = tuneRF()
    X_train, X_test, y_train, y_test=tune.split()
    y_pred=tune.defaultRF(X_train, y_train, X_test, y_test)
    #roc=tune.aucRF(y_test, y_pred)
    #
    randGrid=tune.randomGrid()
    rf_random=tune.trainGrid(X_train, y_train, randGrid)
    tune.evalRand(X_train, y_train, X_test, y_test, rf_random)
    