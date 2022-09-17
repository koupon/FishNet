# import packages
# import streamlit as st

import numpy as np
from collections import Counter

from nltk.corpus import stopwords
import string, re, joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer

# set seed
np.random.seed(419)



# utils
#Adding customized words to default stopwords
new_stopwords = ['How','how','Just','was','Was','coming','Coming','a','Were','were','b','They','c','Can','can', 'Should', 'should','be','Be','d','by','By','Over','over', 'When','when','More','Those','more','those','Both','both','Other','other','Why','why',
    'e','Function', 'f','They','they','g','Js','Often','often','Them','beside','Beside','them','h','What','what''js','i','Which','each','Each','which','j','Don', 'k','l','m','Will', 'will','Through','through','Should','n','o','Now','p','q','r','Id','s','t','Some','some','u','Havin','having','v','w','Can','can','no','No',
    'x','Var','y','Being','It','it','Until','until','being','z','Fjs','A','am','Am','B','Script','C','Document','D','Such','such''E','That','Through','through','From','from','that','F','G','Who','When','when','who','H','Out','out','I','Whom','Above', 'above','whom','J','under','Under','K','That','that','L','M','are','Are','N','Is','is',
    'O','P','Q','he','He','R','S','Them','them','T','She','Through','through','she','W','X','His','his','Y','It','it','Z','Its','While','while','its','into','Meet','meet', 'rk','Ti', 'ng','Vi','JavaScript','Nor','Other','other','nor','Not','not','Me','me','Myself','myself', ' My','my','Yours','your','Ourself',
    'Oureselves','ourself','Ourself', 'Yours','yours','Him','him','Ours','our','Once','once','We','we', 'Again','again','Because','More','more','own','Own','because','Between','between','up','Up','Some','spme','So','same','Same','so','same','Any','any','She','she',
    'PHP', 'php', 'Nor', 'HTML','Not', 'html', 'Only','see','Only', 'go','Owe','owe','Own','get','Same','Under','under','want','So','help','Than', 'make','us','talk','The'
    'also','Too', 'First','Very','name','No','Surname', 'had','Such','Had','Must','must']
 

# turn doc into clean tokens
def cleanDoc(corpus, MAX_LEN=50):
    
    all_corpus = []
    for doc in corpus:
        # split into tokens by white space
        tokens = doc.split()
        
        # turn token to lowercase
        tokens = [x.lower() for x in tokens]
                 
        # prepare regex for char filtering
        re_punc = re.compile('[%s]' % re.escape(string.punctuation)) # remove punctuation from each word
        tokens = [re_punc.sub('', w) for w in tokens]
        
#         tokens = lemmatize_text(' '.join(tokens)).split()
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        
        # filter out stop words
        stop_words = set(stopwords.words('english')+new_stopwords)
        tokens = [w for w in tokens if not w in stop_words]
        
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1][:MAX_LEN]
        
        
        all_corpus.append(' '.join(tokens))
    
    return all_corpus



def tfidf_vectorizer(x):

    # Most frequently occuring bigrams
    # tfidf_vectorizer =  joblib.load("weights/tfidf_vectorizer.pkl")
    tfidf_vectorizer =  joblib.load("weights/tfidf_vectorizer_countvec_and_tfidf.pkl")
    
    return tfidf_vectorizer.transform(x)


def count_vectorizer(x, use_url=None):

    # Most frequently occuring bigrams
    # tfidf_vectorizer =  joblib.load("weights/tfidf_vectorizer.pkl")
    if use_url:
        countvec =  joblib.load("weights/vectorizer.pkl")
    else:
        countvec =  joblib.load("weights/count_vectorizer_countvec_and_tfidf.pkl")
    
    return countvec.transform([x]).todense()

        
def load_model():
    return joblib.load("weights/logclf_countvec_and_tfidf.pkl")

def load_url_model():
    return joblib.load("weights/logclf_url.pkl")


def getLabel(index):
    # key_to_label = ['false', 'true', 'neutral']
    index2label = {0:'bad', 1:'good'}
    return index2label[index].upper()
        

def check_url_format(t):
    if t.startswith('http://') or t.startswith('https://'):
        return t
    else:
        return 'http://'+t


def make_prediction(corpus):
    tfidf_corpus = tfidf_vectorizer(corpus)
    
    result = model.predict(tfidf_corpus).item()
    return getLabel(result)

