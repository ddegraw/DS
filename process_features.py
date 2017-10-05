# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import functools
from collections import Counter

from nltk.corpus import stopwords
#from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.preprocessing import StandardScaler
from string import punctuation
from collections import defaultdict

BASE_DIR = './input/'
GLOVE_DIR = 'D:/Embeddings/Glove/'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_NB_WORDS = 220000
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

def jaccard(row):
    wic = set(row['question1']).intersection(set(row['question2']))
    uw = set(row['question1']).union(row['question2'])
    if len(uw) == 0:
        uw = [1]
    return (len(wic) / len(uw))

def common_words(row):
    return len(set(row['question1']).intersection(set(row['question2'])))

def total_unique_words(row):
    return len(set(row['question1']).union(row['question2']))

def total_unq_words_stop(row, stops):
    return len([x for x in set(row['question1']).union(row['question2']) if x not in stops])

def wc_diff(row):
    return abs(len(row['question1']) - len(row['question2']))

def wc_ratio(row):
    l1 = len(row['question1'])*1.0 
    l2 = len(row['question2'])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def wc_diff_unique(row):
    return abs(len(set(row['question1'])) - len(set(row['question2'])))

def wc_ratio_unique(row):
    l1 = len(set(row['question1'])) * 1.0
    l2 = len(set(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def wc_diff_unique_stop(row, stops=None):
    return abs(len([x for x in set(row['question1']) if x not in stops]) - len([x for x in set(row['question2']) if x not in stops]))

def wc_ratio_unique_stop(row, stops=None):
    l1 = len([x for x in set(row['question1']) if x not in stops])*1.0 
    l2 = len([x for x in set(row['question2']) if x not in stops])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def same_start_word(row):
    if not row['question1'] or not row['question2']:
        return np.nan
    return int(row['question1'][0] == row['question2'][0])

def char_diff(row):
    return abs(len(''.join(row['question1'])) - len(''.join(row['question2'])))

def char_ratio(row):
    l1 = len(''.join(row['question1'])) 
    l2 = len(''.join(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def char_diff_unique_stop(row, stops=None):
    return abs(len(''.join([x for x in set(row['question1']) if x not in stops])) - len(''.join([x for x in set(row['question2']) if x not in stops])))

def word_shares(row):
    q1_list = str(row['question1']).lower().split()
    q1 = set(q1_list)
    q1words = q1.difference(stops)
    if len(q1words) == 0:
        return '0:0:0:0:0:0:0:0'

    q2_list = str(row['question2']).lower().split()
    q2 = set(q2_list)
    q2words = q2.difference(stops)
    if len(q2words) == 0:
        return '0:0:0:0:0:0:0:0'

    words_hamming = sum(1 for i in zip(q1_list, q2_list) if i[0]==i[1])/max(len(q1_list), len(q2_list))

    q1stops = q1.intersection(stops)
    q2stops = q2.intersection(stops)
    
    q1_2gram = set([i for i in zip(q1_list, q1_list[1:])])
    q2_2gram = set([i for i in zip(q2_list, q2_list[1:])])

    shared_2gram = q1_2gram.intersection(q2_2gram)

    shared_words = q1words.intersection(q2words)
    shared_weights = [weights.get(w, 0) for w in shared_words]
    q1_weights = [weights.get(w, 0) for w in q1words]
    q2_weights = [weights.get(w, 0) for w in q2words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
        
    R1 = np.sum(shared_weights) / np.sum(total_weights) #tfidf share
    R2 = len(shared_words) / (len(q1words) + len(q2words)) #count share
    R31 = len(q1stops) / len(q1words) #stops in q1
    R32 = len(q2stops) / len(q2words) #stops in q2
    Rcosine_denominator = (np.sqrt(np.dot(q1_weights,q1_weights))*np.sqrt(np.dot(q2_weights,q2_weights)))
    Rcosine = np.dot(shared_weights, shared_weights)/Rcosine_denominator
    if len(q1_2gram) + len(q2_2gram) == 0:
        R2gram = 0
    else:
        R2gram = len(shared_2gram) / (len(q1_2gram) + len(q2_2gram))
        
    return '{}:{}:{}:{}:{}:{}:{}:{}'.format(R1, R2, len(shared_words), R31, R32, R2gram, Rcosine, words_hamming)
        
def add_word_count(x, df, word):
    x['q1_' + word] = df['question1'].apply(lambda x: (word in str(x).lower())*1)
    x['q2_' + word] = df['question2'].apply(lambda x: (word in str(x).lower())*1)
    x[word + '_both'] = x['q1_' + word] * x['q2_' + word]
 
    
def normalize(df):
    mean_values = df.median(axis=0)
    #df.fillna(mean_values, inplace=True)
    standev = df.std(axis=0)
    df = (df-mean_values)/(standev)    
    return df
#return a feature matrix for auxiliary input
def feat_mat():
    df_train = pd.read_csv(BASE_DIR + 'train.csv')
    df_test  = pd.read_csv(BASE_DIR + 'test.csv')
    df_train = df_train.fillna('empty')
    df_test = df_test.fillna('empty')
    
    df = pd.concat([df_train, df_test])
    df['word_shares'] = df.apply(word_shares, axis=1, raw=True)
    x = pd.DataFrame()


    ques = df[['question1', 'question2']].reset_index(drop='index')
    q_dict = defaultdict(set)
    
    for i in range(ques.shape[0]):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])

    def q1_freq(row):
        return(len(q_dict[row['question1']]))
    
    def q2_freq(row):
        return(len(q_dict[row['question2']]))
    
    def q1_q2_intersect(row):
        return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

    df['q1_q2_intersect'] = df.apply(q1_q2_intersect, axis=1, raw=True)
    df['q1_freq'] = df.apply(q1_freq, axis=1, raw=True)
    df['q2_freq'] = df.apply(q2_freq, axis=1, raw=True)

    x[['q1_q2_intersect', 'q1_freq', 'q2_freq']] = df[['q1_q2_intersect', 'q1_freq', 'q2_freq']]

    df['word_shares'] = df.apply(word_shares, axis=1, raw=True)
    
    x = pd.DataFrame()
    x['word_match']       = df['word_shares'].apply(lambda x: float(x.split(':')[0]))
    x['word_match_2root'] = np.sqrt(x['word_match'])
    x['tfidf_word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[1]))
    x['shared_count']     = df['word_shares'].apply(lambda x: float(x.split(':')[2]))
    
    x['stops1_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[3]))
    x['stops2_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[4]))
    x['shared_2gram']     = df['word_shares'].apply(lambda x: float(x.split(':')[5]))
    x['cosine']           = df['word_shares'].apply(lambda x: float(x.split(':')[6]))
    x['words_hamming']    = df['word_shares'].apply(lambda x: float(x.split(':')[7]))
    x['diff_stops_r']     = x['stops1_ratio'] - x['stops2_ratio']
    
    x['jaccard'] = df.apply(jaccard, axis=1, raw=True) #4
    x['wc_diff'] = df.apply(wc_diff, axis=1, raw=True) #5
    x['wc_ratio'] = df.apply(wc_ratio, axis=1, raw=True) #6
    x['wc_diff_unique'] = df.apply(wc_diff_unique, axis=1, raw=True) #7
    x['wc_ratio_unique'] = df.apply(wc_ratio_unique, axis=1, raw=True) #8

    f = functools.partial(wc_diff_unique_stop, stops=stops)    
    x['wc_diff_unq_stop'] = df.apply(f, axis=1, raw=True) #9
    f = functools.partial(wc_ratio_unique_stop, stops=stops)    
    x['wc_ratio_unique_stop'] = df.apply(f, axis=1, raw=True) #10

    x['same_start'] = df.apply(same_start_word, axis=1, raw=True) #11
    x['char_diff'] = df.apply(char_diff, axis=1, raw=True) #12

    f = functools.partial(char_diff_unique_stop, stops=stops) 
    x['char_diff_unq_stop'] = df.apply(f, axis=1, raw=True) #13

#     X['common_words'] = data.apply(common_words, axis=1, raw=True)  #14
    x['total_unique_words'] = df.apply(total_unique_words, axis=1, raw=True)  #15

    f = functools.partial(total_unq_words_stop, stops=stops)
    x['total_unq_words_stop'] = df.apply(f, axis=1, raw=True)  #16
    
    x['char_ratio'] = df.apply(char_ratio, axis=1, raw=True) #17 
    
    
    x['len_q1'] = df['question1'].apply(lambda x: len(str(x)))
    x['len_q2'] = df['question2'].apply(lambda x: len(str(x)))
    x['diff_len'] = x['len_q1'] - x['len_q2']
    x['caps_count_q1'] = df['question1'].apply(lambda x:sum(1 for i in str(x) if i.isupper()))
    x['caps_count_q2'] = df['question2'].apply(lambda x:sum(1 for i in str(x) if i.isupper()))
    x['diff_caps'] = x['caps_count_q1'] - x['caps_count_q2']
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
    
    sid = SentimentIntensityAnalyzer() 
    
    x['senti1'] = df['question1'].apply( lambda x: sid.polarity_scores(x).get('compound'))
    x['senti2'] = df['question2'].apply( lambda x: sid.polarity_scores(x).get('compound'))
    x['polar'] = (np.sign(x['senti1']) == np.sign(x['senti2'])).astype(int) 

    x.pop('senti1')
    x.pop('senti2')
    
    add_word_count(x, df,'how')
    add_word_count(x, df,'what')
    add_word_count(x, df,'which')
    add_word_count(x, df,'who')
    add_word_count(x, df,'where')
    add_word_count(x, df,'when')
    add_word_count(x, df,'why')
    
    print(x.columns)
    #print(x.describe())

    np.random.seed(1234)
    
    sz_train = df_train.shape[0]
    perm = np.random.permutation(sz_train)
    x_test  = x[sz_train:]
    x_train = x[:sz_train]
    
    idx_train = perm[:int(sz_train*(1-VALIDATION_SPLIT))]
    x_t = pd.concat([x_train.ix[idx_train], x_train.ix[idx_train]])
    
    idx_val = perm[int(sz_train*(1-VALIDATION_SPLIT)):]
    x_val = pd.concat([x_train.ix[idx_val], x_train.ix[idx_val]])
    
    
    x_t = x_t.fillna(0)
    x_test = x_test.fillna(0)
    x_val = x_val.fillna(0)
    
    ss = StandardScaler()
    ss.fit(np.vstack((x_t, x_test,x_val)))
    x_t = ss.transform(x_t)
    x_test = ss.transform(x_test)
    x_val = ss.transform(x_val)
      
   
    return x_t, x_test, x_val
    

#Get Feature Vectors for auxiliary input
aux_train, aux_test, aux_val = feat_mat()


del train_qs
del words
del counts
del stops
#del weights