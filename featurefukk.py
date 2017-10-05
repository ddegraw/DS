# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
#np.random.seed(1337)
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

BASE_DIR = './input/'
GLOVE_DIR = 'D:/Embeddings/Glove/'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_NB_WORDS = 110000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
#Some statistical analysis on the length of tokenized sequences
#mean = 11, std=6, 75%ile = 13, 95%ile = 23, 99%ile = 31, max = 245
MAX_SEQUENCE_LENGTH = 32

def get_weight(count, eps=10000, min_count=2):
    return 0 if count < min_count else 1 / (count + eps)


stops = set(stopwords.words("english"))
train_qs = texts_1 + texts_2 + test_texts_1 + test_texts_2
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

        
def word_shares(row):
    q1 = set(str(row['question1']).lower().split())
    q1words = q1.difference(stops)
    if len(q1words) == 0:
        return '0:0:0:0:0'

    q2 = set(str(row['question2']).lower().split())
    q2words = q2.difference(stops)
    if len(q2words) == 0:
        return '0:0:0:0:0'

    q1stops = q1.intersection(stops)
    q2stops = q2.intersection(stops)

    shared_words = q1words.intersection(q2words)
    shared_weights = [weights.get(w, 0) for w in shared_words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
        
    R1 = np.sum(shared_weights) / np.sum(total_weights) #tfidf share
    R2 = len(shared_words) / (len(q1words) + len(q2words)) #count share
    R31 = len(q1stops) / len(q1words) #stops in q1
    R32 = len(q2stops) / len(q2words) #stops in q2
        
    return '{}:{}:{}:{}:{}'.format(R1, R2, len(shared_words), R31, R32)
        
def add_word_count(x, df, word):
    x['q1_' + word] = df['question1'].apply(lambda x: (word in str(x).lower())*1)
    x['q2_' + word] = df['question2'].apply(lambda x: (word in str(x).lower())*1)
    x[word + '_both'] = x['q1_' + word] * x['q2_' + word]
    
#return a feature matrix for auxiliary input

df_train = pd.read_csv(BASE_DIR + 'train.csv')
df_test  = pd.read_csv(BASE_DIR + 'test.csv')
df_train = df_train.fillna('empty')
df_test = df_test.fillna('empty')
    
    #df_train = df_train[:1000]
    #df_test = df_test[:5000]
    
df = pd.concat([df_train, df_test])
df['word_shares'] = df.apply(word_shares, axis=1, raw=True)
    
x = pd.DataFrame()

x['word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[0]))
x['tfidf_word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[1]))
x['shared_count'] = df['word_shares'].apply(lambda x: float(x.split(':')[2]))
x['stops1_ratio'] = df['word_shares'].apply(lambda x: float(x.split(':')[3]))
x['stops2_ratio'] = df['word_shares'].apply(lambda x: float(x.split(':')[4]))
x['diff_stops_r'] = x['stops1_ratio'] - x['stops2_ratio']

x['len_q1'] = df['question1'].apply(lambda x: len(str(x)))
x['len_q2'] = df['question2'].apply(lambda x: len(str(x)))
x['diff_len'] = x['len_q1'] - x['len_q2']

x['len_char_q1'] = df['question1'].apply(lambda x: len(str(x).replace(' ', '')))
x['len_char_q2'] = df['question2'].apply(lambda x: len(str(x).replace(' ', '')))
x['diff_len_char'] = x['len_char_q1'] - x['len_char_q2']

x['len_word_q1'] = df['question1'].apply(lambda x: len(str(x).split()))
x['len_word_q2'] = df['question2'].apply(lambda x: len(str(x).split()))
x['diff_len_word'] = x['len_word_q1'] - x['len_word_q2']

x['avg_world_len1'] = x['len_char_q1'] / x['len_word_q1']
x['avg_world_len2'] = x['len_char_q2'] / x['len_word_q2']
x['diff_avg_word'] = x['avg_world_len1'] - x['avg_world_len2']

x['exactly_same'] = (df['question1'] == df['question2']).astype(int)
x['duplicated'] = df.duplicated(['question1','question2']).astype(int)
    
'''
    #sid = SentimentIntensityAnalyzer() 
    
    x['senti1'] = df['question1'].apply( lambda x: sid.polarity_scores(x).get('compound'))
    x['senti2'] = df['question2'].apply( lambda x: sid.polarity_scores(x).get('compound'))
    x['polar'] = (np.sign(x['senti1']) == np.sign(x['senti2'])).astype(int) 
    
    del x['senti1']
    del x['senti2']
    
    qtypes = ['who','what','when','where','why','how','which']
    q1type = df[df['question1'] == qtypes]
    q2type = df[df['question2'] == qtypes]
    x['qtype_match'] = sum(q1type*q2type).astype(int)
'''
    
add_word_count(x, df,'how')
add_word_count(x, df,'what')
add_word_count(x, df,'which')
add_word_count(x, df,'who')
add_word_count(x, df,'where')
add_word_count(x, df,'when')
add_word_count(x, df,'why')
    
print('Creating validation set.')
np.random.seed(1234)
perm = np.random.permutation(len(x))
aux_test  = x[df_train.shape[0]:]
x_train = x[:df_train.shape[0]]
    
sz_train = len(x_train)
    
idx_train = perm[:int(sz_train*(1-VALIDATION_SPLIT))]
aux_train = pd.concat([x_train.ix[idx_train], x_train.ix[idx_train]])
    
idx_val = perm[int(sz_train*(1-VALIDATION_SPLIT)):]
aux_val = pd.concat([x_train.ix[idx_val], x_train.ix[idx_val]])
   
del train_qs
del words
del counts
del stops


