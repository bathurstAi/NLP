# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 02:28:35 2019

@author: kishite
"""
import pandas as pd
import numpy as np

import time
import datetime

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

import collections

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn import decomposition, ensemble

from yellowbrick.text import FreqDistVisualizer

no_features = 1000

max_epochs = 100
vec_size = 30
alpha = 0.025

#    in_dir = "C:\Users\kishite\Documents\Education\Queens\MMAI\MMAI891\Project\Ppython\DataOutput"
#    out_dir = "C:\Users\kishite\Documents\Education\Queens\MMAI\MMAI891\Project\Ppython"

class Vec():
    
    model = 0
    """
    Read csv
    """
    def readCorp(self, fname):
        df = pd.read_csv(fname, encoding = "iso-8859-1")
        return (df)
    
    """
    Tag/Label docs
    """
    def tagged(self, docs):
        documents = [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(docs)]
        return(documents)
    
    """
    Create count of vectors
    """
    def tf(self, df):
        tf_vectorizer = CountVectorizer(min_df=.05, max_df=.5, max_features=no_features, ngram_range=[1,3])
        dtm_tf = tf_vectorizer.fit_transform(df['descriptions'])
        print(dtm_tf.shape)

        df = pd.DataFrame(dtm_tf.toarray(), columns=tf_vectorizer.get_feature_names())
        print(df.head())
        
        #Show top tokens
        # Calculate column sums from DTM
        sum_words = dtm_tf.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in tf_vectorizer.vocabulary_.items()]
        
        # Now, sort them
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        
        # Display top few
        print(words_freq[:20])
        
        #Visualize Freq. of top 25 tokens
        plt.figure(figsize=(5,8))
        visualizer = FreqDistVisualizer(features=tf_vectorizer.get_feature_names(), n=25)
        visualizer.fit(dtm_tf)
        visualizer.poof()
        
    
    """
    Doc2Vec model
    """
    
    def doc2vec(self, docs):   
        documents = [TaggedDocument(words=doc.split(), 
                                    tags=[str(i)]) for i, doc in enumerate(docs)]
     
        
        self.model = Doc2Vec(documents, vector_size=vec_size, 
                        #window=2, 
                        min_count=2, 
                        #workers=4, 
                        #epochs=10000, 
                        alpha=alpha,
                        min_alpha=0.00025,
                        #seed=123,
                        dm=1)
       # model.build_vocab(documents)
        print(self.model)
        print("Length: ", len(documents))
        for i in range(0, len(documents)):
            print(self.model.docvecs[i])
            
       
        
        for epoch in range(max_epochs):
            print('iteration {0}'.format(epoch))
            self.model.train(documents,
                        total_examples=self.model.corpus_count,
                        epochs=self.model.iter)
            # decrease the learning rate
            self.model.alpha -= 0.0002
            # fix the learning rate, no decay
            self.model.min_alpha = self.model.alpha
            
        self.model.save("d2v.model")
        print("Model Saved") 
        
        return(self.model)
    
    def test(self, obj):
        obj= Doc2Vec.load("d2v.model")
        #to find the vector of a document which is not in training data
        test_data = word_tokenize("I love chatbots".lower())
        v1 = obj.infer_vector(test_data)
        print("V1_infer", v1)
        
        # to find most similar doc using tags
        similar_doc = obj.docvecs.most_similar('1')
        print("sim: ", similar_doc)


        # to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
        print("obj: ",obj.docvecs['1'])   
    
    """
    Assessing the model
    """
    def assessing(self, obj, tag):
        ranks = []
        second_ranks = []
        for doc_id in range(len(tag)):
            inferred_vector = obj.infer_vector(tag[doc_id].words)
            sims = obj.docvecs.most_similar([inferred_vector], topn=len(obj.docvecs))
            rank = [docid for docid, sim in sims].index(str(doc_id))
            print("R: ", rank)
            ranks.append(rank)
    
            second_ranks.append(sims[1])    

            collections.Counter(ranks)             
    """
    Split dataset into train and test
    """
    def split(self, df):
        # split the dataset into training and validation datasets 
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df)

        # label encode the target variable 
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        valid_y = encoder.fit_transform(valid_y)
        
    """
    Matrix of words to vector
    """
    def mapping(self, obj, tag):
        # pair word and vector together 
        embeddings_index = {}
        for i in range(len(tag)):
            print("I:", i)
            print("Length: ", len(tag[i].words))
            for j in range(len(tag[i].words)):
                print("J: ", j)
                embeddings_index[tag[i].words[j]] = obj.docvecs[i][j]
                print("eb: ", embeddings_index)
                
        
            embedding_matrix = np.zeros((len(tag) + 70))
            for word in tag[i].words:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
                print("Matrix: ", embedding_matrix)                
        print("M: ", embedding_matrix)
        
    """
    TF-IDF
    """
    # word level tf-idf
    # word level tf-idf
#    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
#    tfidf_vect.fit(trainDF['text'])
#    xtrain_tfidf =  tfidf_vect.transform(train_x)
#    xvalid_tfidf =  tfidf_vect.transform(valid_x)
#    
#    # ngram level tf-idf 
#    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
#    tfidf_vect_ngram.fit(trainDF['text'])
#    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
#    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
#    
#    # characters level tf-idf
#    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
#    tfidf_vect_ngram_chars.fit(trainDF['text'])
#    xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
#    xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 
    
if __name__ == "__main__":
    pd.set_option('display.max_colwidth', -1)
    vector = Vec()
    df = vector.readCorp(r'C:\Users\kishite\Documents\Education\Queens\MMAI\MMAI891\Project\Ppython\DataOutput\export_dataframe_all.csv')  
    vector.tf(df)
    tagged=vector.tagged(df['descriptions'])
    #print("T: ", tagged)
    model2=vector.doc2vec(df['descriptions'])
    vector.test(model2)
    #vector.assessing(model2, tagged)
    vector.mapping(model2, tagged)
    
