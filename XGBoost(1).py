
# coding: utf-8

# In[1]:


import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

best_xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.01,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=1000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)


# In[2]:


import pandas as pd

df= pd.read_csv('BGIS_Vendor_1hot_feature_LDA.csv')




X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25, random_state=0)


# In[4]:


best_xgb_model.fit(X_train,y_train)
y_pred = best_xgb_model.predict(X_test)
print("XGBoost Regressor Score : ", best_xgb_model.score(X_test, y_test))
#print("Random Forest RMSE : ", sqrt(mean_squared_error(y_test,y_pred)))
print("XGBoost  MAE : ",mean_absolute_error(y_test,y_pred))

