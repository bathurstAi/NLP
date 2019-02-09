# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 19:10:19 2019

@author: kishite
"""


import os
import re
import nltk

import unidecode

import pandas as pd
import numpy as np

import spacy
import sys

from os import listdir, makedirs
from os.path import exists, join
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import WordPunctTokenizer 

from nltk.util import ngrams

#from ftfy import fix_encoding

#from translate_api.translate_api import api
#from googletrans import Translator

from pattern.en import spelling
from pattern.en import suggest

from nltk.corpus import wordnet



class Res():
    
    """
    Initialize spaCy
    """
    nlp = spacy.load('en', disable=['parser', 'ner'])
    
    """
    spaCy lemmitization
    """
    def spCy(self, text):
        doc = self.nlp(text)
        return ([token.lemma_ for token in doc])
    
    """
    Reduces repeated characters in word
    """
    def reduce_lengthening(self, tokens):
        pattern = re.compile(r"(.)\1{2,}")
        return pattern.sub(r"\1\1", tokens)

    """
    Checks for spelling
    """
    def sp(self, tokens):
        correct_word = spelling(tokens) 
        return(correct_word)
        
    """
    Correct spelling
    """  
    def corr(self, text):
        word = suggest(text)
        max_word=max(word[:][1])
        return max_word[0]

#    def translate(self, raw):
#         for index, row in raw.iterrows():
#            # REINITIALIZE THE API
#            translator = Translator()
#            try
#                translated = translator.translate(row, dest='en')
#                raw.set_value(index, 'Text', translated.text)
#            except Exception as e:
#                print(str(e))
#                continue
#        return(raw)
        

    def makedirs(self, directory):
        """
            Checks if directory doesn't exist, then creates it.

            Paras:
                directory: name of directory

            Returns:
                Boolean
        """
        if not exists(directory): 
            makedirs(directory)
            return True
        else:
            return False

    def pd_csv(self, path):
        """
            Converts CSV to pandas Dataframe

            Paras:
                path: pfath to csv sheet
            
            Return:
                df: extracted dataframe
        """
        return pd.read_csv(path, encoding = 'utf-8')
    
    def pd_xlsx(self, path):
        """
            Converts XLSX to pandas Dataframe

            Paras:
                path: path to xlsx sheet
            
            Return:
                df: extracted dataframe
        """
        print("GGGG")
        return pd.read_excel('Data_Cleaned.xlsx')

    def remove_columns(self, dataframe, column_names):
        """
            Filters out unwanted columns

            Paras:
                dataframe: dataframe to filter
                column_names: columns to keep

            Return:
                df: filtered dataframe
        """
        try:
            dataframe = dataframe[column_names]
            dataframe = dataframe.dropna()
            return dataframe.reset_index(drop = True)
        except (ValueError, KeyError):
            pass

    def concate_columns(self, column_1, column_2):
        """
            Concates columns and adds the strings together.

            Paras:
                column_1, column_2: columns to concatenate
            Returns:
                returns concatenated list of strings
        """
        return [str(column_1[i] + ". " + column_2[i]) for i in range(len(column_1))]

    def merge_workbooks(self, dataframes, column_names):
        """
            Merges multiple workbooks based on column names

            Paras:
                dataframes: dfs to merge
                Column_names: column names to merge on
            Returns:
                df: concatenated dataframes
        """
        return pd.concat([self.remove_columns(df, column_names) for df in dataframes])

    def download_nltk_tools(self, path):
        """
            Downloads NLTK tools

            Paras: 
                path: path to download tools
            Returns:
            None
        """
        if self.makedirs(path) == True:
            nltk.download("punkt", path)
            nltk.download("stopwords", path)
            nltk.download("wordnet", path)
            nltk.data.path.append(path)
            os.remove(join(path, "corpora/stopwords.zip"))
            os.remove(join(path, "corpora/wordnet.zip"))
            os.remove(join(path, "tokenizers/punkt.zip"))
            
    """
    tokenize sentence into work tokens
    """
    def token_nize(self, tokens):
        words = word_tokenize(str(tokens))
        print("T:", words)
        return(words)
        
    """
    Normalize word Lower all capitilization
    """
    def lower(self, tokens):
        low = [word.lower() for word in tokens]
        print("hter")
        print("L:", low)
        return(low)
        
        
    """
    remove all numbers in sentence
    """
    def numbers(self, sentence):
        sentence = re.sub(r'\d+', '', sentence)
         
    """
    remove all characters in sentence
    """
    def characters(self, sentence):
        sentence = re.sub(r'[^\w\s]', '', sentence)
        
    """
    remove all unicode characters in sentence
    """
    def ucode(self, sentence):
         unidecode.unidecode(sentence)
    
    """
    remove all characters in sentence
    """
    # With Porter Stemmer
    def portStem(self, token):
        ps = PorterStemmer()
        for w in token:
            print(ps.stem(w))

    """
    remove all characters in sentence
    """
    # With Lancaster Stemmer
    def lanStem(self, token):
        ls = LancasterStemmer()
        for w in token:
            print(ls.stem(w))
    """
    remove all characters in sentence
    """
    # With SnballStemmer("english")
    def snowStem(self, token):
        ss = SnowballStemmer()
        for w in token:
            print(ss.stem(w))
    
    """
    Detrmine unique words
    """  
    def unique(self, raw):      
        return(raw.unique())
        
    """
    Analyze structure of text
    """
    # Structure analysis
    def stucAnalysis(self, token):
        num_words = token.apply(lambda x: len(x.split()))
        num_words_mean, num_words_std = np.mean(num_words), np.std(num_words)
        print("Number of words: ", num_words)
        print("Words mean: ", num_words_mean)
        print("Std. dev. of words: ", num_words_std)

        num_sentences = token.apply(lambda x: len(re.split( '~ ...' ,'~'.join(x.split('.')))))
        num_sentences_mean = np.mean(num_sentences)
        print("Number of sentences: ", num_sentences_mean)
        
        
    """
    Select number of ngrams
    """
    def nGrams(self, tokens, num=2):
        grams = ngrams(tokens,num)
        return [grams for grams in grams]
    
        
    """
    Select only english text
    """
    def english(self, tokens):
            if wordnet.synsets(tokens):
                word = tokens
                print("Word: ", word)
            return word 
    
    """
    Get worknet POS tag for lemmitization (select NOUNS)
    """
    def getWordnetPos(self, word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

        
    def remove_contractions(self, raw):
        """
            Removes contractions to clean sentences
            
            Paras:
                raw: raw text data
            Returns:
                raw: cleaned text
        """
        contractions = { 
                        "ain't": "is not",
                        "aren't": "are not",
                        "can't": "cannot",
                        "could've": "could have",
                        "couldn't": "could not",
                        "didn't": "did not",
                        "doesn't": "does not",
                        "don't": "do not",
                        "hadn't": "had not",
                        "hasn't": "has not",
                        "haven't": "have not",
                        "he'd": "he would",
                        "he'll": "he will",
                        "he's": "he is",
                        "how'd": "how did",
                        "how'll": "how will",
                        "how's": "how is",
                        "I'd": "I would",
                        "I'll": "I will",
                        "I'm": "I am",
                        "I've": "I have",
                        "isn't": "is not",
                        "it'd": "it would",
                        "it'll": "it will",
                        "it's": "it is",
                        "let's": "let us",
                        "ma'am": "madam",
                        "mayn't": "may not",
                        "might've": "might have",
                        "mightn't": "might not",
                        "must've": "must have",
                        "mustn't": "must not",
                        "needn't": "need not",
                        "o'clock": "of the clock",
                        "oughtn't": "ought not",
                        "shan't": "shall not",
                        "sha'n't": "shall not",
                        "she'd": "she would",
                        "she'll": "she will",
                        "she's": "she is",
                        "should've": "should have",
                        "shouldn't": "should not",
                        "shouldn't've": "should not have",
                        "so've": "so have",
                        "so's": "so as",
                        "that'd": "that would",
                        "that's": "that is",
                        "there'd": "there had",
                        "there's": "there is",
                        "they'd": "they would",
                        "they'll": "they will",
                        "they're": "they are",
                        "they've": "they have",
                        "to've": "to have",
                        "wasn't": "was not",
                        "we'd": "we would",
                        "we'll": "we will",
                        "we're": "we are",
                        "we've": "we have",
                        "weren't": "were not",
                        "what'll": "what will",
                        "what're": "what are",
                        "what's": "what is",
                        "what've": "what have",
                        "when's": "when is",
                        "when've": "when have",
                        "where'd": "where did",
                        "where's": "where is",
                        "where've": "where have",
                        "who'll": "who will",
                        "who'll've": "who will have",
                        "who's": "who is",
                        "who've": "who have",
                        "why's": "why has",
                        "why've": "why have",
                        "will've": "will have",
                        "won't": "will not",
                        "won't've": "will not have",
                        "would've": "would have",
                        "wouldn't": "would not",
                        "y'all": "you all",
                        "you'd": "you had / you would",
                        "you'll": "you will",
                        "you'll've": "you will have",
                        "you're": "you are",
                        "you've": "you have"
                    }
        if raw in contractions:
            for key, value in contractions.items():
                raw = re.sub(key, value, raw)
                return raw
        else:
            return raw

    def clean_text(self, text, short = True, length = True, contra = True, remove_stopwords = True, lemmatize = True, english = False, ngrams = False, spelling = True, spCy=False):
        print("cleantoo")
       # print("T:", text)
        """
            Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings
            check spelling, lemmatize and compare with wordnet corpus for english words
            Paras:
                text: text data to clean
                remove_stopwords: if true, remove stop words  text to reduce noise
                lemmatize: if true lemmatizes word
                english: if true compares w/ wordnet corpus to keep only english words
                ngrams: if true creates ngrams 
            Returns:
                text: cleaned text data
        """
        if contra:
            print("CLEAN")
            text = [self.remove_contractions(word) for word in sent_tokenize(text.lower())]
            text = " ".join(text)
    
            text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
            text = re.sub(r'\<a href', ' ', text)
            text = re.sub(r'&amp;', '', text) 
            text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
            text = re.sub(r'<br />', ' ', text)
            text = re.sub(r'\'', ' ', text)
            text = re.sub(r'[^a-zA-Z]', " ", text)
        
        if length:
            print("LENGTH")
        #text = text.split()
            text = self.reduce_lengthening(text)# for w in text]
        #text = " ".join(text)
        
        if spelling:
           print("SPELLING")
           text = text.split()
           word = suggest(text)
           max_word=max(word[:][1])
           #for w in self.token_nize(text)]
           text = " ".join(max_word[0])
        
        if remove_stopwords:
            print("STOP")
            text = text.split()
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
            text = " ".join(text)
            
        if short:
           print("SHORT")
           text = ' '.join([w for w in text.split() if len(w)>3])
           print("S: ", text)          
        
        if lemmatize:
            print("LEM")
            text_sent = nltk.word_tokenize(text)
            text = [WordNetLemmatizer().lemmatize(w, self.getWordnetPos(w)) for w in text_sent]
            text = " ".join(text)
        
        # Spacy Lemmtization
        if spCy:
            text = " ".join(self.spCy(text))
        
            
        if english:
            print("ENGLISH")
            text = ' '.join([w for w in text.split() if wordnet.synsets(w)])
            print("P: ", text)
            
        if ngrams:
            print("NGRAM")
            text = text.split()
            text = [self.nGrams(text)]
            #text = " ".join(text) 
            
        return text

    def save_data(self, directory, name, docs, mode = "w"):
        """
            Saves data to directory

            Paras:
                directory: directory to save data
                name: name of file
            Returns:
                None
        """
        self.makedirs(directory)
        with open(join(directory, name), mode, encoding = "utf-8") as file:
            file.write(docs)

    def filterReview(self, df, name, value):
        """
            Filter dataframe by returning rows with column criteria
            Paras:
                df: dataframe
                name: name of column
                value: value to filter based on
            Return:
                df: data frame
        """
        return df.loc[df[name] == value]

    def loadData(self, path, column_names):
        """
            Loads the csvs in prepation for processing
            Paras:
                path: path
                column_names: columns to merge data on
            Return:
                df: merged dataframes
        """
        print("here3")
        #files = [doc for doc in listdir(path) if doc.endswith(".csv")]
        files = [doc for doc in listdir(path) if doc.endswith(".xlsx")]
        print(files)
        dataframes = [self.pd_xlsx(join(path, sheet)) for sheet in files]
        if len(dataframes)>1: return self.merge_workbooks(dataframes, column_names)
        else: return dataframes[0]

    def saveResults(self, df):
        df.to_csv("TrainingResults.csv", encoding = 'utf-8', index = False)

    def performance(self, df):
        tp = len(self.filterReview(df, "status", "correct"))
        fp = len(self.filterReview(df, "status", "incorrect"))
        precision = round(float(tp/(tp+fp)), 2)*100
        accuracy = round(float(tp/len(df)), 2)*100
        print("Precision: ", precision)
        print("Accuracy: ", accuracy)
