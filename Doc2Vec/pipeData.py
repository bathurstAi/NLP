# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 22:37:13 2019

@author: Everard Rodney
"""

import re
import pandas as pd

from pathos.multiprocessing import ProcessingPool as Pool

from resources import Res

"""
    Module to process text data 
            
"""

class Pipe():

    def __init__(self):
        """
            Initalizes DataProcessing class with utilities and parallel processing
            
        """
        self.res = Res()
        self.pool = Pool()

    def getDescription(self, description):
    
        """
            Retuns list of cleaned description from dataframe list column
            Paras:
                summaries: list of work order desciptions
            Returns:
                summaries: list of cleaned descriptions
        """
        description = [re.sub(r"[^a-zA-Z]", " ", description[i].lower()) for i in range(len(description))]
        return list(self.pool.map(self.res.clean_text, description))

    def createDataframe(self, description):
        """
            Creates dataframe class of cleaned descriptions.
            
        """
        return pd.DataFrame({"descriptions": description})

    def ProcessData(self, column_names = ["Description_Document"]):
        """
            Runs DataProcessing class
            
        """
        # Look in directory for dataset 
        dataframe = self.res.loadData(r'./Final/Data', column_names)
        description = self.getDescription(list(dataframe["Description_Document"]))
        return self.createDataframe(description)

if __name__ == "__main__":
############### CALL TO PRE-PROCESS DATA AND WRITE TO CSV ################################
    pd.set_option('display.max_colwidth', -1)
    Preprocessing = Pipe()
    dataframe = Preprocessing.ProcessData()
    print("df: ", dataframe)
    export_csv = dataframe.to_csv (r'\DataPre\bgis_vendorPre_words.csv', index = None, header=True)
  
  
  