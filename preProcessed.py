# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 19:43:39 2019

@author: kishite
"""
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import WordPunctTokenizer 

#import nltk

import csv
import unidecode

import re
from collections import Counter


#from nltk.tokenize import PunktSentenceTokenizer

import pickle

from pattern.en import suggest

text=[]
# tokenize sentence 
def token_nize(text):
    words = word_tokenize(str(text))
    print("T:", words)
    return(words)
    
filtered_sentence = []
stop_words = set(stopwords.words("english"))

def stop_words(text):
    filtered_sentence.clear()
    for w in text:
        if w != stop_words:
            filtered_sentence.append(w)
    #print("filtered:", filtered_sentence)

    save_stop_words = open("pickle/stop_words.pickle","wb")
    pickle.dump(filtered_sentence, save_stop_words)
    save_stop_words.close()
    print("fl:",filtered_sentence)        
    return(filtered_sentence)
    
#Normalize word Lower all capitilization
def lower(text):
   low = [word.lower() for word in text]
   print("hter")
   print("L:", low)
   return(low)
   
sp_chk = []
#Spelling Checking
def sp(text):
    print("sp")
    sp_chk.append(suggest(text))
    return()
    
#POS tagging
def pos_tagging(words):
    print("pos")
    tagged = pos_tag(words)
    print(tagged)
    return(tagged)
 
lemmatize_list=[]
lemmatizer = WordNetLemmatizer()
#Lemmintization
def lemmitize(pos):
    print("lemm")
    lemmatize_list.clear()
    #words = word_tokenize(str(text))
    words = pos
    #print("words: ", words)  
    for w in words:
        lem_word = lemmatizer.lemmatize(w)
        lemmatize_list.append(lem_word)

    save_lemmatize_words = open("pickle/lemmatize.pickle","wb")
    pickle.dump(lemmatize_list, save_lemmatize_words)
    save_lemmatize_words.close()

    return(lemmatize_list)
    
# With Porter Stemmer
def portStem(token):
    ps = PorterStemmer()
    for w in token:
        print(ps.stem(w))

# With Lancaster Stemmer
def lanStem(token):
    ls = LancasterStemmer()
    for w in token:
        print(ls.stem(w))
 
# With SnballStemmer("english")
def snowStem(token):
    ss = SnowballStemmer()
    for w in token:
        print(ss.stem(w))
    
"""
Equivalent method with TreebankWordTokenizer
"""
def treebankWordToken(text):
    tokenizer = TreebankWordTokenizer()
    print("\nEquivalent method with TreebankWordTokenizer \n", tokenizer.tokenize(text))


"""
Equivalent method with WordPunctTokenizer  
"""
def wordPuncToken(text):
    tokenizer = WordPunctTokenizer()
    print("\nEquivalent method with WordPunctTokenizer \n", tokenizer.tokenize(text))
 
"""
Remove blancs on words
"""
def remove_blanc(tokens):
    tokens = [token.strip() for token in tokens]
    print(tokens)
    return(tokens)
    
"""
Removing accent marks and other diacritics - before tokens words
"""

def remove_accent_before_tokens(sentences):
    res = unidecode.unidecode(sentences)
    return(res)
    print("After removing accent markes before tokenize words : \n", res)

"""
Removing accent marks and other diacritics - After tokens words
"""

def remove_accent_after_tokens(tokens):
    tokens = [unidecode.unidecode(token) for token in tokens]
    return(tokens)
    print("After removing accent markes ", tokens)

"""
Method 1 : Using the brown corpus in NLTK and "in" operator
"""
from nltk.corpus import brown
word_list = brown.words()
len(word_list)

word_set = set(word_list)
"looked" in word_set

"""
Method 2 : Peter Norvig sur un seul mot
"""



def words(text): return re.findall(r'\w+', text.lower())
WORDS = Counter(words(open('../input/big.txt').read()))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
    
"""
Exemple avec des mots au hasard 
"""
print(correction('speling'))
print(correction('fial'))
print(correction("misstkaes"))
 
def preprocessing(text):
    
    #tokenize
    token=token_nize(text)
    # remove words less than three letters
    
    # lower capitalization
    low = lower(text)
    
    #    spell = sp(low)
    
     # remove stopwords
    stop = stop_words(low)
       

#    spell = sp(low)
    
    #pos
    pos=pos_tagging(token)
    
    # lemmatize
    lmtzr = WordNetLemmatizer(pos)
    print(lmtzr)

     

#read excel file 
#df=pd.read_excel('Data_Cleaned.xlsx')
#csv_reader = csv.reader('Data_Cleaned.csv', delimiter=',')
#next(cs#v_reader, None)  # skip header
df = pd.read_excel('Data_Cleaned.csv')
#csv_reader
print("hd")
print(df)
#convet dataframe to numpy array
text = df.iloc[:, 10].as_matrix()
print("Arr:", text)
preprocessing(text)
