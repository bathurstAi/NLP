# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:48:01 2019

@author: Everard Rodney
"""

import pandas as pd
import numpy as np

#from resources import Res
#from pipeData import Pipe
#from featureEng import Vec
#from createData import Data


import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from numba import cuda
from numba import vectorize

"""
        Module Used to split dataset, fit to model and run with regression model.
        Used Random Forest and Grid Search to Hypertune model.
        Used Cross Validation in Grid Search and regression.
"""

RANDOM_SEED = 7

class classifierNLP():
    
    def __init__(self):
    
        """
        Initalize resources, preprcess and feature engineering classes
        """
#        self.res = Res()
#        self.preprocess = Pipe()
#        self.feature = Vec()
#        self.data = Data()
        
        #collections.Counter(ranks)    
         
    """
    Read csv
    """
    def readCorp(self, fname):
        df = pd.read_csv(fname, encoding = "iso-8859-1")
        return (df)
    
    """
    Read file
    """
    def readfile(self, fname):
        
        df = self.readCorp(fname)
        
        # Split data into training and target variables
        #Section to data and label
        data_X = df.iloc[:, -1].values #LDA FEATURE SET
        data_Y = df.iloc[:, 537].values #LDA FEATURE SET
        #data_X = df.iloc[:, -1].values #TF FEATURE SET
        #data_Y = df.iloc[:, 580].values #TF FEATURE SET
        #data_X = df.iloc[:, :-1].values #TF_IDF FEATURE SET
        #data_Y = df.iloc[:, 1527].values #TF_IDF FEATURE SET
        return(data_X, data_Y)  
        
    """
    Split dataset into train and test
    """
    def split(self):
        
        ####### Uncomment to determine which feature-set to read into model ########################3
        df = self.readCorp(r'\DataFeat\BGIS_Vendor_1hot_feature_DOC2VEC.csv')
        #df = self.readCorp(r'\DataFeat\BGIS_Vendor_1hot_feature_LDA.csv')
        #df = self.readCorp(r'\DataFeat\BGIS_Vendor_1hot_feature_TF.csv')
        #df = self.readCorp(r'\DataFeat\BGIS_Vendor_1hot_feature_TF_LDA.csv')
        #df = self.readCorp(r'\DataFeat\BGIS_Vendor_1hot_feature_TFIDF.csv')
        #df = self.readCorp(r'\DataFeat\BGIS_Vendor_1hot_feature_TFIDF_LDA.csv')
        #df = self.readCorp(r'\DataFeat\BGIS_Vendor_1hot_feature_TFIDF_TF_LDA.csv')
        #df = self.data.readCorp(r'DataFeatureSet\BGIS_Vendor_1hot_feature_all3.csv')
        print(df.shape)
        # Split data into training and target variables
        #Section to data and label
        #data_X = df.iloc[:, :-1].values #LDA FEATURE SET
        #data_Y = df.iloc[:, 537].values #LDA FEATURE SET
        #data_X = df.iloc[:, :-1].values #TF FEATURE SET
        #data_Y = df.iloc[:, 580].values #TF FEATURE SET
        #data_X = df.iloc[:, :-1].values #TF_IDF FEATURE SET
        #data_Y = df.iloc[:, 1528].values #TF_IDF FEATURE SET
        #data_X = df.iloc[:, :-1].values #TF_IDF_LDA FEATURE SET
        #data_Y = df.iloc[:, 1537].values #TF_IDF_LDA FEATURE SET
        #data_X = df.iloc[:, :-1].values #TF_LDA FEATURE SET
        #data_Y = df.iloc[:, 588].values #TF_LDA FEATURE SET
        data_X = df.iloc[:, :-1].values # DOC2VEC FEATURE SET
        data_Y = df.iloc[:, 5784].values #DOC2VEC FEATURE SET
        #print("X:", data_X)
        #print("Y:", data_Y)
        # split the dataset into training and validation datasets 
        #Split dataset into training and testing set
        X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size = 0.2, random_state=RANDOM_SEED)
        #train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df)

        # label encode the target variable 
#        encoder = preprocessing.LabelEncoder()
#        y_train = encoder.fit_transform(y_train)
#        y_test = encoder.fit_transform(y_test)
        
        return ( X_train, X_test, y_train, y_test)
    
#   
#    @vectorize(['float32(float32, float32, float32, float32)'], target='cuda')    
    def rFor(self, Xtrain, ytrain, X_test, y_test):
#        
        model=RandomForestRegressor(n_jobs=-1)
        model.fit(Xtrain, ytrain)
        print("HERE")
#        feature_imp = pd.Series(clf.feature_importances_,index=iris.feature_names).sort_values(ascending=False)
#        feature_imp
        # Try different numbers of n_estimators - this will take a minute or so
        estimators = np.arange(10, 1000, 10)
        scores = []
        for n in estimators:
            model.set_params(n_estimators=n)
            model.fit(Xtrain, ytrain)
            scores.append(model.score(X_test, y_test))
        plt.title("Effect of n_estimators")
        plt.xlabel("n_estimator")
        plt.ylabel("score")
        plt.plot(estimators, scores)
        
        predictions = model.predict(X_test)
        errors = abs(predictions - y_test)
        mae = np.mean(errors)
        
        # K-fold CV
        # Perform K-Fold CV
        scores = cross_val_score(model, Xtrain, ytrain, cv=10, scoring='neg_mean_absolute_error')
        predictions = cross_val_predict(model, Xtrain, ytrain, cv=10)
        errors=abs(predictions-y_test)
        mae = np.mean(errors)
       
        return(predictions, mae)
        
    
    """
    Use GridSearch to determine the max-depth and number of estimators
    """
    #@vectorize(['float32(float32, float32)'], target='cuda')
    def rfr_model(self,Xtrain, ytrain, X_test, y_test):

        # Perform Grid-Search
        gsc = GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid={
                'max_depth': range(3,7),
                'n_estimators': (10, 50, 100, 1000),
            },
            cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
        
        grid_result = gsc.fit(ytrain, ytrain)
        
        rfr = RandomForestRegressor(max_depth=grid_result.best_params_best_["max_depth"], n_estimators=grid_result.best_params_["n_estimators"], random_state=False, verbose=False)
    
#        # Perform K-Fold CV
        scores = cross_val_score(rfr, Xtrain, ytrain, cv=10, scoring='neg_mean_absolute_error')
        predictions = cross_val_predict(rfr, Xtrain, ytrain, cv=10)
        errors=abs(predictions-y_test)
        mae = np.mean(errors)
        return (scores, predictions, mae)
        
      


if __name__ == "__main__":
############### Split into training and validation set and call Random Forest model #####################    
    classNLp = classifierNLP()
    X_train, X_test, y_train, y_test=classNLp.split()
#    model=classNLp.rFor(X_train, y_train)
#    classNLp.tst_model(model, X_train, y_train, X_test, y_test)
#    data_X, data_Y = classNLp.readfile(r'DataFeatureSet\export_feature_all.csv')
    #assert list[classNLp.vec_rFor(X_train, y_train, X_test, y_test)]
    pred, mae = classNLp.rFor(X_train, y_train, X_test, y_test)
    scores, pred2, mae2 = classNLp.rFor.rfr_model(X_train, y_train, X_test, y_test)
    
    print(pred)
    print(mae)
    print(pred2)
    print(mae2)
