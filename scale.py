# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 21:55:44 2019

@author: kishite
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

df_raw = pd.read_excel(r'bgis_vendor_MMAI891.xlsx')
df = pd.read_excel(r'bgis_vendor_MMAI891.xlsx')
sc = StandardScaler()

#df.describe()
#df.isnull().sum().sum()
#Remove Work_Duration_Days_Rounded and off_by_days_Rounded, and work order description
#df=df.drop('Work_Duration_Days_Rounded', axis=1)
#df=df.drop('off_by_days_Rounded', axis=1)
#df=df[df['Func_Burdened_Cost'] <= 1000]
#df=df[df['Func_Burdened_Cost'] >= 20]
df=df.drop('Description_Document', axis=1)

#df = df.drop('Building_ID',axis = 1)
df = df.drop('Property_Usage',axis = 1)
df = df.drop('Province',axis = 1)
# Drop column as it is encoded
df = df.drop('Region_Name',axis = 1)
# Drop column as it is encoded
df = df.drop('ServiceType_Cd',axis = 1)
# Drop column as it is encoded
df = df.drop('ServiceProvider_Type',axis = 1)
# Drop column as it is encoded
df = df.drop('WorkOrderSource_Cd',axis = 1)
# Drop column as it is encoded
df = df.drop('WorkOrderType_Cd',axis = 1)
# Drop column as it is encoded
df = df.drop('WorkOrder_Priority_Desc',axis = 1)
# Drop column as it is encoded
df = df.drop('City_up2',axis = 1)
#Drop target
df2=df.drop('Func_Burdened_Cost', axis = 1)

#Scale

scaled_df = sc.fit_transform(df2)
scaled_df = pd.DataFrame(scaled_df, columns=['Rentable_sqft', 'Estimated_Time_Days', 'LeaseInd2', 'Work_Duration_Days_Rounded', 'off_by_days_Rounded', 'doc_lengths', 'WorkOrder_Priority_Desc'])

#### Get one hot encoding of columns Property_Usage
one_hot = pd.get_dummies(df_raw['Property_Usage'])
## Join the encoded df
scaled_df = scaled_df.join(one_hot)
#### Get one hot encoding of columns Province
one_hot = pd.get_dummies(df_raw['Province'])
## Join the encoded df
scaled_df = scaled_df.join(one_hot)
#
## Join the encoded df
#df = df.join(one_hot)
#### Get one hot encoding of columns Region_Name
#one_hot = pd.get_dummies(df_raw['Region_Name']
## Join the encoded df
#f = df.join(one_hot)
#### Get one hot encoding of columns ServiceType_Cd
one_hot = pd.get_dummies(df_raw['ServiceType_Cd'])
## Join the encoded df
scaled_df = scaled_df.join(one_hot)
#### Get one hot encoding o_f columns ServiceProvider_Type
one_hot = pd.get_dummies(df_raw['ServiceProvider_Type'])
## Join the encoded df
scaled_df = scaled_df.join(one_hot)
#### Get one hot encoding of columns WorkOrderSource_Cd
one_hot = pd.get_dummies(df_raw['WorkOrderSource_Cd'])
## Join the encoded df
scaled_df = scaled_df.join(one_hot)
#### Get one hot encoding of columns City_up2
one_hot = pd.get_dummies(df_raw['City_up2'])
## Join the encoded df
scaled_df = scaled_df.join(one_hot)
#### Get one hot encoding of columns WorkOrderType_Cd
one_hot = pd.get_dummies(df_raw['WorkOrderType_Cd'])
## Join the encoded df
scaled_df = scaled_df.join(one_hot)
#### Get one hot encoding of columns WorkOrder_Priority_Desc
one_hot = pd.get_dummies(df_raw['WorkOrder_Priority_Desc'])
## Join the encoded df
scaled_df = scaled_df.join(one_hot)
#### Get one hot encoding of columns Building_ID
#one_hot = pd.get_dummies(df_raw['Building_ID'])

#df_scaled = pd.DataFrame(df_scaled)
## Join the encoded df
#df_scaled = pd.concat(one_hot, axis=1)
scaled_df['Description_Document'] = df_raw['Description_Document']
scaled_df['Func_Burdened_Cost'] = df_raw['Func_Burdened_Cost']
#columns= df_raw['Description_Document']
#for col in columns:
#    df_scaled[col] = one_hot[col]
#    
#df_scaled
scaled_df=scaled_df[scaled_df['Func_Burdened_Cost'] <= 1000]
scaled_df=scaled_df[scaled_df['Func_Burdened_Cost'] >= 20]


#join labels
#df_scaled_labeled = pd.DataFrame(df_scaled, columns = df_raw.columns)
#df_scaled=df_raw['Description']

#print(df.isnull().sum())


# Drop column Building_ID as it is encoded
#type(one_hot)




#join target
#df_scaled_labeled.index = df.index
#df_scaled_labeled_target=df_scaled.join(df[['Func_Burdened_Cost']])

#df_scaled_labeled.Func_Burdened_Cost = df.Func_Burdened_Cost.astype(float)
#Export
#np.savetxt(r'C:\Users\ANTHONYPitfield\Documents\MMAI\MMAI891 NLP\Term Project\BGIS_Vendor_scaled1hot.csv',df_scaled_labeled_target,delimiter=',')
scaled_df.to_csv(r'C:\Users\kishite\Documents\Education\Queens\MMAI\MMAI891\Project\Ppython\Final\Data\BGIS_Vendor_scaled1hot.csv')
