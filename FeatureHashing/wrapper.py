# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 17:46:51 2019

@author: Muhammad Shahbaz
"""
#import lda2vec as l2v
import numpy as np
import pandas as pd
from hashing import HashingEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 
from sklearn.ensemble import RandomForestRegressor



dfbgisnlp = pd.read_csv("./Final/DataPre/bgis_vendorPre_words")
dfbgiscost = pd.read_csv("./Final/Data/BGIS_Vendor_scaled1hot.csv")
#Removing Outliers
from sklearn.feature_extraction import FeatureHasher
fh = FeatureHasher(n_features=10, input_type='string')
desc = dfbgisnlp["descriptions"]
#len(dfbgisnlp)
result=fh.fit(desc)
resultin = []
for item in range(len(desc)):
    #resultin.append(result.transform(item))
    #resultin.append(result.toarray())
    print(desc)
df = pd.DataFrame(result.toarray(), columns=['fh1', 'fh2', 'fh3', 'fh4', 'fh5', 'fh6', 'fh7', 'fh8','fh9','fh10'])
from sklearn.datasets import load_boston
dfboston = load_boston()
import pandas as pd
from sklearn.datasets import load_boston
bunch = load_boston()
y = dfbgiscost["Func_Burdened_Cost"]
X = dfbgisnlp 
enc = HashingEncoder(cols=['descriptions']).fit(X, y)
numeric_dataset = enc.transform(X)
print(numeric_dataset.info())

#Prediction
X_train, X_test, y_train, y_test = train_test_split(numeric_dataset,y,test_size=.25, random_state=0)
reg = RandomForestRegressor()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
y_pred = lin_reg.predict(X_test)

print("Random Forest Regressor Score : ", lin_reg.score(X_test, y_test))
#print("Random Forest RMSE : ", sqrt(mean_squared_error(y_test,y_pred)))
print("Random Forest MAE : ",mean_absolute_error(y_test,y_pred))



    
