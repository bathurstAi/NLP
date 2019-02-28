# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 02:28:35 2019

@author: Everard Rodney
"""
import pandas as pd
import numpy as np

from itertools import chain

from pipeData import Pipe

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

## Maximum number of features
no_features = 1000

## Train for 100 epochs
max_epochs = 1
## Vector size is 30 because documents have max. terms between 20 and 30
vec_size = 30
## Initial Learning rate
alpha = 0.025

#    in_dir = "C:\Users\kishite\Documents\Education\Queens\MMAI\MMAI891\Project\Ppython\DataOutput"
#    out_dir = "C:\Users\kishite\Documents\Education\Queens\MMAI\MMAI891\Project\Ppython"

"""
    Vectoriztion module used for Doc2Vec, TF and TF-IDF
"""
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
        
        tf_vectorizer = CountVectorizer(min_df=0.01, max_df=0.85, max_features=no_features, ngram_range=[2,3])
        dtm_tf = tf_vectorizer.fit_transform(df['descriptions'])
        print("dtm:", dtm_tf.shape)

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
        return(df, sum_words, words_freq)
        
    
    """
    Doc2Vec model
    """
    
    def doc2vec(self, docs):   
        documents = [TaggedDocument(words=doc.split(), 
                                    tags=[str(i)]) for i, doc in enumerate(docs)]
     
        
        self.model = Doc2Vec(documents, vector_size=vec_size, 
                        window=6, 
                        min_count=0.20, 
                        workers=8, 
                        #epochs=1000, 
                        alpha=alpha,
                        min_alpha=0.00025,
                        #seed=123,
                        max_vocab_size=2000,
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

###### Test to infer similiar documents in corpus ################## 
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
            print("Sims: ", sims)
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
        return(train_x, valid_x, train_y, valid_y)
        
    """
    Matrix of words to vector
    """
    
    def mapping(self, obj, tag):
        # pair word and vector together 
        embeddings_index_all={}
        
#        embedding_matrix = np.zeros((len(tag)))
#        for i in range(len(tag)):
#            words = tag[i].words
#            embeddings_index_all = dict.fromkeys(words, 0)
        
        for i in range(len(tag)):
            embeddings_index = {}
           # words = tag[i].words
#            print("L:", len(list(chain.from_iterable(tag[i][i][i][:]))))
#            for j in range(lehn(list(chain.from_iterable(tag[i][i][i][:])))-1):
            
            for j in range(len(tag[i].words)):
                embeddings_index[tag[i].words[j]] = obj.docvecs[i][j]
                embeddings_index_all[i]=embeddings_index
#                if(obj.docvecs[i][j] != 0):
#                    embeddings_index[tag[i].words[j]] = obj.docvecs[i][j]
#                    #embeddings_index[i][j] = obj.docvecs[i][j]
#                else:
#                    print("Words:", len(tag[i].words))
#                    embeddings_index[tag[i].words[j]] = 0
            
        #print("counter", i)
#        print("eb: ", embeddings_index.items())
#        print("eb2: ", embeddings_index)
#        print("all: ", embeddings_index_all)
#        print("LLL2", len(embeddings_index_all))
#        embedding_vector = {}
#        for word in tag[i].words:
#                embedding_vector = embeddings_index_all.get(word)
#                if embedding_vector is not None:
#                    print("ebm:", embedding_vector)
#                    embedding_matrix[i] = embedding_vector
        #df = pd.DataFrame.from_dict(embeddings_index_all, orient='index')
#        embedding_matrix.toarray()
#        print("LLL", len(embedding_matrix))
        df_matrix = pd.DataFrame(embeddings_index_all)
        df_matrix = df_matrix.transpose()
        df_matrix.fillna(0, inplace=True)
       
        df_matrix[df_matrix == 0].count()
       
        return(df_matrix)        
        
    """
    Create Matrix of vector/words    
    """
    def matrix(self, fname):    
        vector = Vec()
        df = vector.readCorp(fname)  
        print("dfM: ", df)
        vector.tf(df)
        tagged=vector.tagged(df['descriptions'])
        model2=vector.doc2vec(df['descriptions'])
        matrix=vector.mapping(model2, tagged)
        return(matrix.fillna(0, inplace=True))
    
    """
    Count vectors for frequency count of words in document corpus   
    """
    def cntVec(self, df):  
        # create a count vectorizer object 
        count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        count_vect.fit(df)
        
        bow_df = pd.DataFrame(count_vect.toarray(), columns=count_vect.get_feature_names(), index=df.index)
        bow_df.shape
#        #Split dataset
#        train_x, valid_x, train_y, valid_y= self.split(self, df)
#        
#        # transform the training and validation data using count vectorizer object
#        xtrain_count =  count_vect.transform(train_x)
#        xvalid_count =  count_vect.transform(valid_x)
        return(bow_df)
        
    """     
    TF-IDF
    
    """
    def tf_idf(self, df):
                        
        #Split dataset
#        train_x, valid_x, train_y, valid_y= self.split(df)
        # word level tf-idf
        # word level tf-idf
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=no_features)
        tfidf_vect.fit(df)
        print("3:", tfidf_vect)
        xtrain_tfidf =  tfidf_vect.transform(df)
        print("Train: ", xtrain_tfidf)
        idf_df = pd.DataFrame(xtrain_tfidf.toarray(), columns=tfidf_vect.get_feature_names(), index=df.index)
        idf_df.shape
        pd.set_option('display.max_colwidth', -1)
        print("idf1: ", idf_df)
#        xvalid_tfidf =  tfidf_vect.transform(valid_x)
        
        # ngram level tf-idf 
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(4,4), max_features=no_features)
        tfidf_vect_ngram.fit(df)
        print("2:", tfidf_vect_ngram)
        xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(df)
        print("Train ngram: ", xtrain_tfidf_ngram)
        #%time dtm_tf = tf_vectorizer.fit_transform(kiva_df['en_clean'])       
        idf_df_ngram = pd.DataFrame(xtrain_tfidf_ngram.toarray(), columns=tfidf_vect.get_feature_names(), index=df.index)
        idf_df_ngram.shape
        pd.set_option('display.max_colwidth', -1)
        print("idf: ", idf_df_ngram) 
#        xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
        
        # characters level tfc-idf
        tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(1,2), max_features=no_features)
        tfidf_vect_ngram_chars.fit(df)
        print("1:", tfidf_vect_ngram_chars)
        xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(df)
        print("Train ngram_char: ", xtrain_tfidf_ngram_chars)
#        xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)
        return(idf_df_ngram)# Change from idf_df, idf_df_ngram for different results
if __name__ == "__main__": 
########## DIRECT CALL TO PRE-PROCESS DATA ######
    pd.set_option('display.max_colwidth', -1)
    Preprocessing = Pipe()
    df_p = Preprocessing.ProcessData()
    vector = Vec()
    
    #df_csv = pd.read_excel(r'C:\Users\kishite\Documents\Education\Queens\MMAI\MMAI891\Project\Ppython\Final\DataPre\bgis_vendorPre_words.xlsx')

    #df_p.shape
    #df = vector.readCorp(r'C:\Users\kishite\Documents\Education\Queens\MMAI\MMAI891\Project\Ppython\DataOutPre\bgis_vendorPre.csv')  
    
#############CALL TO TF AND TF-IDF VECTORIZATION AND WRITE TO CSV ###################################
    #df_tf, su, freq = vector.tf(df_p)  
    #df =_ df.reindex(df_tf.index.dropna())
    #df_tfidf=vector.tf_idf(df_p['descriptions'])
    #df_tf.to_csv (r'\TF\TF.csv', index = None, header=True)
    #df_tfidf.to_csv (r'\TF_IDF\TF_IDF.csv', index = None, header=True)
    
############## TAG DATASET AND CALL DOC2VEC ###################################
    tagged=vector.tagged(df_p['descriptions'])
    model2=vector.doc2vec(df_p['descriptions'])
    
########### USED TO TEST AND ASSESS THE MODEL #####################
    #vector.test(model2)
    #vector.assessing(model2, tagged)

############# CALL TO MAP DOC2VEC VECTORS TO MATRIX ####################    
    #xtr, xval, xtrngram, xvalngram, xtrchar, cvalchar = vector.tf_idf(df)
    np.set_printoptions(threshold=np.inf)
    matrix=vector.mapping(model2, tagged)

############ ATTEMPT TO REDUCE MATRIX DIMENSTIONS BY ELIMINATING TERMS ############    
   # matrix[matrix == 0].count()
#    matrix[matrix > 0].count()
#    
#    for col in matrix.columns:
#        cnt=matrix[matrix[col]>0].count()
#        if ( cnt <6):
#            
#            matrix.drop(col, inplace=True, axis=1)
#        print(matrix[matrix[col]>0].count())
#        
#    matrix_drop = matrix.drop([matrix[matrix < 6].count()])
#    plt.plot(matrix[matrix > 0].count(), color = 'red', label = 'Non - Zero Count')
#    
#    plt.title('Count of Non-zero')
#    plt.legend()
#    plt.show()
    #matrix.fillna(0, inplace=True)
    #matrix.shape
    print(matrix)
    export_csv = matrix.to_csv (r'\DataOutMatrix\bgis_matrix_words_param.csv', index = None, header=True)
#    