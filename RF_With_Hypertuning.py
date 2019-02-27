
# coding: utf-8

# In[9]:


import pandas as pd

df= pd.read_csv('BGIS_Vendor_1hot_feature_TFIDF_TF_LDA.csv')




X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[10]:


y.head()


# In[16]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, GridSearchCV
gsc = GridSearchCV(
estimator=RandomForestRegressor(),
param_grid={
            'max_depth': (125,150,175,200,225,250),
            'n_estimators': (250,300,350,400,450,500),
           },
           cv=5, scoring='neg_mean_absolute_error', verbose=0, n_jobs=-1)
grid_result = gsc.fit(X, y)
#  

#rfr = RandomForestRegressor(max_depth=grid_result.best_params_best_["max_depth"], n_estimators=grid_result.best_params_["n_estimators"], random_state=False, verbose=False)


# In[17]:


grid_result.best_params_


# In[19]:


from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25, random_state=0)

#

#Using Random Forest Regressor
reg = RandomForestRegressor(n_estimators=1000, max_depth=125)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print("Random Forest Regressor Score : ", reg.score(X_test, y_test))
#print("Random Forest RMSE : ", sqrt(mean_squared_error(y_test,y_pred)))
print("Random Forest MAE : ",mean_absolute_error(y_test,y_pred))


# In[ ]:


Random Forest Regressor Score :  0.320793966371
Random Forest MAE :  138.154111577

