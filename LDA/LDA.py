import pandas as pd 
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim import models, corpora
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
import gensim 
from gensim.models import CoherenceModel
#import optimal_k 
import matplotlib.pyplot as plt

#from gensim import models, corpora

### BGIS cleaned text
data = pd.read_csv(r'BGIS_Vendor_scaled1hot.csv',encoding = "ISO-8859-1")
##Preprocessed text
text_df = pd.read_csv(r'\DataPre\bgis_vendorPre_words.csv',encoding = "ISO-8859-1")

text_df =text_df['descriptions'].tolist()


# Build a Dictionary - association word to numeric id
porter = PorterStemmer()
STOPWORDS = nltk.corpus.stopwords.words('english')
newStopWords = ['please','replace',"name","thank"]
STOPWORDS.extend(newStopWords)

def clean_text(text):
    tokenized_text = word_tokenize(text)#.lower
    cleaned_text = porter.stem(text)
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    return cleaned_text

# For gensim we need to tokenize the data and filter out stopwords
tokenized_data = []
for text in text_df:
    #print(text)
    tokenized_data.append(clean_text(str(text)))

bigram = gensim.models.Phrases(tokenized_data, min_count=5, threshold=100) # higher threshold fewer phrases.
bigram_mod = gensim.models.phrases.Phraser(bigram)
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

# Form Bigrams
data_words_bigrams = make_bigrams(tokenized_data)  


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values 

################################## LDA ###############################################################################

# Build a Dictionary - association word to numeric id
dictionary = corpora.Dictionary(data_words_bigrams)   
# Transform the collection of texts to a numerical form
corpus = [dictionary.doc2bow(text) for text in tokenized_data]
# Human readable format of corpus (term-frequency)
[[(dictionary[id], freq) for id, freq in cp] for cp in corpus[:1]]


 
#Finding optimal number of topics
model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=data_words_bigrams, start=2, limit=40, step=6)
# Show graph
limit=40
start=2
step=6
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# Build the LDA model
NUM_TOPICS = 8 #optium = 8
lda_model = models.LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)

# # Build the LSI model
# lsi_model = models.LsiModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)

print("LDA Model:")
for idx in range(NUM_TOPICS):
    # Print the first 10 most representative topics
    print("Topic #%s:" % idx, lda_model.print_topic(idx))


from pyLDAvis import gensim
import pyLDAvis
visualisation = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')



# create lda values 
lda_value = []
for token in tokenized_data:
    #print(text)
    bow = dictionary.doc2bow(token)
    lda_value.append(lda_model[bow])

bow = dictionary.doc2bow(clean_text(text_df[3]))
print(lda_model[bow])
bow3 = dictionary.doc2bow(tokenized_data[3])
print(lda_model[bow3])
    
##### USED TO MAP LDA WEIGHTED TOPICS TO CSV #########################################    
embeddings_index_all = {}

for i in range(len(lda_value)):
     embeddings_index = {}
     for j in range(len(lda_value[i])):
         embeddings_index[lda_value[i][j][0]] = lda_value[i][j][1]
         embeddings_index_all[i]=embeddings_index
         print("ind: ", i, j)
pd.set_option('display.max_colwidth', -1)
df_lda = pd.DataFrame.from_dict(embeddings_index_all, orient='index')
        
df_lda.fillna(0, inplace=True)
df_lda.head()
df_lda.tail()
df_lda.to_csv (r'C:\LDA.csv', index = None, header=True)
