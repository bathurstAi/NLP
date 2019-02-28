# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 07:46:35 2019

@author: Everard Rodney
"""


import pandas as pd
import numpy as np

from resources import Res
from pipeData import Pipe
from featureEng import Vec

"""
			Creates Feature set by concatenating features from different models with BGIS cleaned dataset 
			that contain features related to cost
            Output:  csv file that is fed into Machine Learning Model.  In our case that would be Random Forest and a shallow Neural Network
""" 

class Data():
    
    """
    	Initalizes resource class with feature engineering
        as:
		None
        urns:
		None
	"""
    def __init__(self):

        self.res = Res()
        self.vector = Vec()
        self.pipe = Pipe()
        
    """
    Read csv
    """
    def readCorp(self, fname):
        df = pd.read_csv(fname, encoding = "iso-8859-1")
        return (df)
    
    """
			Creates training data set with labels appended to beginging of each label

			Paras:
				df: datafframe
			Returns:
				None
	""" 
    def createTrainingCorpus(self, dfVec, fname):#fanme2, fname3): # Uncomment when adding additional nlp features
		
        df = self.readCorp(fname)
        #df2 = self.readCorp(fname2)
        #df3 = self.readCorp(fname3)
        df_feature = pd.concat([df, dfVec], axis=1)#df2, df3, axis=1) Uncomment and delete for additional features
        df_feature.fillna(0, inplace=True)
        df_feature.to_csv (r'C:\DataFeat\BGIS_Vendor_1hot_feature_DOC2VEC.csv', index = None, header=True)
        #df_feature.to_csv (r'C:\DataFeat\BGIS_Vendor_1hot_feature_LDA.csv', index = None, header=True)
        #df_feature.to_csv (r'C:\DataFeat\BGIS_Vendor_1hot_feature_TF.csv', index = None, header=True)
        #df_feature.to_csv (r'C:\DataFeat\BGIS_Vendor_1hot_feature_TF_LDA.csv', index = None, header=True)
        #df_feature.to_csv (r'C:\DataFeat\BGIS_Vendor_1hot_feature_TFIDF.csv', index = None, header=True)
        #df_feature.to_csv (r'C:\DataFeat\BGIS_Vendor_1hot_feature_TFIDF_LDA.csv', index = None, header=True)
        #df_feature.to_csv (r'C:\DataFeat\BGIS_Vendor_1hot_feature_TFIDF_TF_LDA.csv', index = None, header=True)
        return(df_feature)
        
    """
			Runs DataSets class and creates training set

			Paras: 
				None
			Returns:
				None
	"""	
    def createSet(self,fnamePre, fname):
		
        #df = self.vector.matrix(fnamePre)
        df = self.readCorp(fnamePre)
        print("Pre:", df)
        df_feat=self.createTrainingCorpus(df, fname)
        return(df_feat)
        
if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    data = Data()
##############  Used to create desired dataset and combination of nlp methods for Machine Learning model and regression analysis #################
    #df_LDA=data.createSet(r'Final\Data\BGIS_Vendor_scaled1hot_wo_description.csv', r'Final\LDA\LDA.csv')
    #df_TF=data.createSet(r'Final\Data\BGIS_Vendor_scaled1hot_wo_description.csv', r'Final\TF\TF.csv'_)
    #df_TF_IDF=data.createSet(r'Final\Data\BGIS_Vendor_scaled1hot_wo_description.csv', r'Final\TF_TDF\TF_IDF.csv')
    #df_TF_LDA=data.createSet(r'Final\Data\BGIS_Vendor_scaled1hot_wo_description.csv', r'Final\LDA\LDA.csv', r'Final\TF\TF.csv')
    #df_TF_IDF_LDA=data.createSet(r'Final\Data\BGIS_Vendor_scaled1hot_wo_description.csv', r'Final\LDA\LDA.csv', r'Final\TF_IDF\TF_IDF.csv')
    #df_TF_TF_IDF_LDA=data.createSet(r'Final\Data\BGIS_Vendor_scaled1hot_wo_description.csv', r'Final\LDA\LDA.csv', , r'Final\TF\TF.csv, r'Final\TF_IDF\TF_IDF.csv')
    df_F_DOC2VEC=data.createSet(r'Final\Data\BGIS_Vendor_scaled1hot_wo_description.csv', r'Final\doc2vec\bgis_matrix_words_param.csv')
    np.isnan(df_F_DOC2VEC.values.any())
    df_F_DOC2VEC.shape
        
        
        
        
    
    