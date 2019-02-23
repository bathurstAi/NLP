# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:48:01 2019

@author: kishite
"""

import pandas as pd

import numpy as np

from resources import Res
from pipeData import Pipe
from featureEng import Vec
from createData import Data

import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.initializers import Constant

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from keras.models import Sequential
from keras.layers import Dropout
from sklearn.ensemble import RandomForestClassifier


import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import xgboost, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from numba import cuda
from numba import vectorize



RANDOM_SEED = 7

class classifierNLP():
    
    def __init__(self):
    
        """
        Initalize resources, preprcess and feature engineering classes
        """
        self.res = Res()
        self.preprocess = Pipe()
        self.feature = Vec()
        self.data = Data()
        
        #collections.Counter(ranks)    

    """
    Read file
    """
    def readfile(self, fname):
        
        df = self.data.readCorp(fname)
        
        # Split data into training and target variables
        #Section to data and label
        data_X = df.iloc[:, -1].values #LDA FEATURE SET
        data_Y = df.iloc[:, 537].values #LDA FEATURE SET
        return(data_X, data_Y)  
        
    """
    Split dataset into train and test
    """
    def split(self):
        
        df = self.data.readCorp(r'Final\DataFeat\BGIS_Vendor_1hot_feature_LDA.csv')
        #df = self.data.readCorp(r'DataFeatureSet\BGIS_Vendor_1hot_feature_all3.csv')
        
        # Split data into training and target variables
        #Section to data and label
        data_X = df.iloc[:, :-1].values #LDA FEATURE SET
        data_Y = df.iloc[:, 537].values #LDA FEATURE SET
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
    
    """     
    TF-IDF
    
    """
    def tf_idf(self, df):
        
        #Split dataset
#        train_x, valid_x, train_y, valid_y= self.split(df)
        # word level tf-idf
        # word level tf-idf
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
        tfidf_vect.fit(df)
        print("3:", tfidf_vect)
#        xtrain_tfidf =  tfidf_vect.transform(train_x)
#        xvalid_tfidf =  tfidf_vect.transform(valid_x)
        
        # ngram level tf-idf 
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,4), max_features=5000)
        tfidf_vect_ngram.fit(df)
        print("2:", tfidf_vect_ngram)
#        xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
#        xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
        
        # characters level tfc-idf
        tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,4), max_features=5000)
        tfidf_vect_ngram_chars.fit(df)
        print("1:", tfidf_vect_ngram_chars)
#        xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
#        xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)
        #return(xtrain_tfidf, xvalid_tfidf,xtrain_tfidf_ngram, xvalid_tfidf_ngram, xtrain_tfidf_ngram_chars, xvalid_tfidf_ngram_chars)
#    def setParameters(self, lr = None, lrUpdateRate = None, dim = None, ws = None, epoch = None, neg = None, loss = None, thread = None, saveOutput = None):
#        """
#            Sets parameters to train NN
#
#            Paras:
#                -lr                 learning rate [0.05]
#                -lrUpdateRate       change the rate of updates for the learning rate [100]
#                -dim                size of word vectors [100]
#                -ws                 size of the context window [5]
#                -epoch              number of epochs [5]
#                -neg                number of negatives sampled [5]
#                -loss               loss function {ns, hs, softmax} [ns]
#                -thread             number of threads [12]
#                -pretrainedVectors  pretrained word vectors for supervised learning []
#                -saveOutput         whether output params should be saved [0]
#            Returns:
#                training parameters
#        """
#        if lr == None: lr = " "
#        else: lr = "-lr %s " %lr
#        
#        if lrUpdateRate == None: lrUpdateRate = " "
#        else: lrUpdateRate = "-lrUpdateRate %s " %lrUpdateRate
#        
#        if dim == None: dim = " "
#        else: dim = "-dim %s " %dim
#        
#        if ws == None: ws = " "
#        else: ws = "-ws %s " %ws
#        
#        if epoch == None: epoch = " "
#        else: epoch = "-epoch %s " %epoch
#        
#        if neg == None: neg = " "
#        else: neg = "-neg %s " %neg
#        
#        if loss == None: loss = " "
#        else: loss = "-loss %s " %loss
#        
#        if thread == None: thread = " "
#        else: thread = "-thread %s " %thread
#        
#        if saveOutput == None: saveOutput = " "
#        else: saveOutput = "-saveOutput %s " %saveOutput
#        
#        return lr + lrUpdateRate + dim + ws + epoch + neg + loss + thread + saveOutput

#    def trainClassifier(self, hyper_parameters):
#        """
#            Trains supervised classifier
#            Paras:
#                hyper_parameters: parameters to train neural net
#            Returns:
#                None
#        """
#        self.utls.makedirs("./fastTextModels")
#        system("./fastText/fasttext supervised -input ./Dataset/training_processed/training.txt -output ./fastTextModels/model_1 -label __label__ {}").format(hyper_parameters)
       
#     def neural():
#         model = Sequential()
#         embedding_layer=Enbedding(num_words),
#                                    EMBEDDING_DIM, 
#                                    embeddings_initalizer=Constan(embedding_matrix)
#                                    input_length=max_length,
#                                    trainable=False)
#        model.add(embedding_layer)
#        model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
#        model.add(Dense(1, activation='sigmoid'))
#        
#        #try using different optimizers and different optimizer configs
#        model.compile(loss='binary_crossentropy', optimier='adam', metrics=['mae'])

#    def naive(self, xtrain_count, xtrain_tfidf, xtrain_tfidf_ngram, xtrain_tfidf_ngram_chars):
#        # Naive Bayes on Count Vectors,
#        accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
#        print( "NB, Count Vectors: ", accuracy)
#        
#        # Naive Bayes on Word Level TF IDF Vectors
#        accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
#        print( "NB, WordLevel TF-IDF: ", accuracy)
#        
#        # Naive Bayes on Ngram Level TF IDF Vectors
#        accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
#        print( "NB, N-Gram Vectors: ", accuracy)
#        
#        # Naive Bayes on Character Level TF IDF Vectors
#        accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
#        print( "NB, CharLevel Vectors: ", accuracy)
#        
#    def linear(self, xtrain_count, xtrain_tfidf, xtrain_tfidf_ngram, xtrain_tfidf_ngram_chars):
#        # Linear Classifier on Count Vectors
#        accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
#        print ("LR, Count Vectors: ", accuracy)
#        
#        # Linear Classifier on Word Level TF IDF Vectors
#        accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
#        print ("LR, WordLevel TF-IDF: ", accuracy)
#        
#        # Linear Classifier on Ngram Level TF IDF Vectors
#        accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
#        print ("LR, N-Gram Vectors: ", accuracy)
#        
#        # Linear Classifier on Character Level TF IDF Vectors
#        accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
#        print ("LR, CharLevel Vectors: ", accuracy)
#        
#    def SVM(self, xtrain_tfidf_ngram):
#       # SVM on Ngram Level TF IDF Vectors
#        accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
#        print ("SVM, N-Gram Vectors: ", accuracy) 
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
#        scores = cross_val_predict(model, Xtrain, y, cv=10, scoring='neg_mean_absolute_error')
#        predictions = cross_val_predict(rfr, Xtrain, ytrain, cv=10)
#        errors=abs(predictions-y_test)
#        mae = np.mean(errors)
       
        return(predictions, mae)
        
    
    """
    Use GridSearch to determine the max-depth and number of estimators
    """
#    @vectorize(['float32(float32, float32)'], target='cuda')
#    def rfr_model(self, X, y):
#
#        # Perform Grid-Search
#        gsc = GridSearchCV(
#            estimator=RandomForestRegressor(),
#            param_grid={
#                'max_depth': range(3,7),
#                'n_estimators': (10, 50, 100, 1000),
#            },
#            cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
#        
#        grid_result = gsc.fit(X, y)
#        
#        rfr = RandomForestRegressor(max_depth=grid_result.best_params_best_["max_depth"], n_estimators=grid_result.best_params_["n_estimators"], random_state=False, verbose=False)
#    
#        # Perform K-Fold CV
#        scores = cross_val_predict(rfr, X, y, cv=10, scoring='neg_mean_absolute_error')
#        predictions = cross_val_predict(rfr, X, y, cv=10)
#        return (scores, predictions)
        
      


if __name__ == "__main__":
    
    classNLp = classifierNLP()
    X_train, X_test, y_train, y_test=classNLp.split()
#    model=classNLp.rFor(X_train, y_train)
#    classNLp.tst_model(model, X_train, y_train, X_test, y_test)
#    data_X, data_Y = classNLp.readfile(r'DataFeatureSet\export_feature_all.csv')
    #assert list[classNLp.vec_rFor(X_train, y_train, X_test, y_test)]
    pred, mae = classNLp.rFor(X_train, y_train, X_test, y_test)
    
    pred
    mae
