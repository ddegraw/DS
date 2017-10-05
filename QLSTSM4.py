# -*- coding: utf-8 -*-
import os
import csv
import codecs
import numpy as np
import pandas as pd
#np.random.seed(1337)
import gc
import re
from collections import Counter
import itertools

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from string import punctuation

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, merge, LSTM, Dropout, Bidirectional, Embedding, Activation
#from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten, Lambda
from keras.models import Model
#from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras import backend as K
#import sys

#sys.setdefaultencoding('utf-8')
re_weight = True
BASE_DIR = './input/'
GLOVE_DIR = 'D:/Embeddings/Glove/'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
#MAX_SEQUENCE_LENGTH = 230
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
#Some statistical analysis on the length of tokenized sequences
#mean = 11, std=6, 75%ile = 13, 95%ile = 23, 99%ile = 31, max = 245
MAX_SEQUENCE_LENGTH = 32

print('Indexing word vectors.')
embeddings_index = {}
f = codecs.open(os.path.join(GLOVE_DIR, 'glove.840B.300d.txt'), encoding='utf_8')
for line in itertools.islice(f,0,5000):
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

print('Indexing word vectors.')
stops = set(stopwords.words("english"))

print('Processing text dataset')

def text_to_wordlist(text, remove_stopwords=True, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        #stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

def get_weight(count, eps=10000, min_count=2):
    return 0 if count < min_count else 1 / (count + eps)
        
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
def feat_mat():
    df_train = pd.read_csv(BASE_DIR + 'train.csv')
    df_test  = pd.read_csv(BASE_DIR + 'test.csv')
    df_train = df_train.fillna('empty')
    df_test = df_test.fillna('empty')
    
    df_train = df_train[:1000]
    df_test = df_test[:5000]
    
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
    
    x_train = x[:df_train.shape[0]]
    x_test  = x[df_train.shape[0]:]
    
    x_train.fillna(0)
    x_test.fillna(0)   
    
    return x_train, x_test
    
texts_1 = [] 
texts_2 = []
labels = []

#with codecs.open(TRAIN_DATA_FILE, encoding='utf_8',errors="ignore") as f:
with codecs.open(TRAIN_DATA_FILE) as f:
    #reader = csv.reader(f, delimiter=',')
    df = pd.read_csv(f, delimiter=',', nrows = 1000)
    #texts_1.append(text_to_wordlist(values[3].decode('utf-8')))
    texts_1= df['question1'].tolist()
    texts_2 = df['question2'].tolist()
    labels = df['is_duplicate'].apply(int).tolist()
print('Found %s texts.' % len(texts_1))

test_texts_1 = []
test_texts_2 = []
test_labels = []  # list of label ids
#with codecs.open(TEST_DATA_FILE, encoding='utf_8',errors="ignore") as f:
with codecs.open(TEST_DATA_FILE) as f:
    #reader = csv.reader(f, delimiter=',')
    df = pd.read_csv(f, delimiter=',', nrows = 5000)
    #test_texts_1.append(text_to_wordlist(values[1].decode('utf-8')))
    test_texts_1 = df['question1'].tolist()
    test_texts_2 = df['question2'].tolist()
    test_labels = df['test_id'].tolist()
print('Found %s texts.' % len(test_texts_1))

#texts_1 = [s.encode('utf_8') for s in texts_1]
#texts_2 = [s.encode('utf_8') for s in texts_2]
#test_texts_1 = [s.encode('utf_8') for s in test_texts_1]
#test_texts_2 = [s.encode('utf_8') for s in test_texts_2]
#test_labels = [s.encode('utf_8') for s in test_labels]

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)

sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_labels = np.array(test_labels)

train_qs = texts_1 + texts_2
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

del test_sequences_1
del test_sequences_2
del sequences_1
del sequences_2
del texts_1
del texts_2
del test_texts_1
del test_texts_2

gc.collect()

print('Preparing embedding matrix.')
# prepare embedding matrix
nb_words = min(MAX_NB_WORDS, len(word_index))

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    word = word.decode('utf-8','ignore')
    if i >= nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None    

#Get Feature Vectors for auxiliary input
aux_train, aux_test = feat_mat()

# Model Architecture #
#num_lstm = np.random.randint(int(MAX_SEQUENCE_LENGTH*0.75), int(MAX_SEQUENCE_LENGTH*1.25))
num_lstm = 32
num_dense = 64
#num_dense = np.random.randint(int(0.75*num_lstm/2.0), int(1.25*num_lstm/2.0))
rate_drop_lstm = 0.15 + np.random.rand() * 0.10
rate_drop_dense = 0.15 + np.random.rand() * 0.10

act = 'relu'
re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'lstm_aux_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, rate_drop_dense)

embedding_layer = Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
                            
#shared_lstm = Bidirectional(LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))
shared_lstm = Bidirectional(LSTM(num_lstm))

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='input1')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = shared_lstm(embedded_sequences_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32',name='input2')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = shared_lstm(embedded_sequences_2)

merged_lstm = merge([x1,y1], mode='concat')
merged_lstm = Dropout(rate_drop_dense)(merged_lstm)
merged_lstm = BatchNormalization()(merged_lstm)

auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(merged_lstm)

auxiliary_input = Input(shape=(aux_train.shape[1],),name='aux_input')
foo = Activation('linear')(auxiliary_input)

merged = merge([merged_lstm,foo],mode='concat')
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)
merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

final = Dense(1, activation='sigmoid', name='main_output')(merged)

train = aux_train.as_matrix()

model = Model(input=[sequence_1_input,sequence_2_input,auxiliary_input], output=[final,auxiliary_output])
model.compile(loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'}, optimizer='nadam', metrics=['acc'], loss_weights=[1., 0.2])
print(STAMP)

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit({'input1': data_1, 'input2': data_2, 'aux_input': train}, {'main_output':labels, 'aux_output':labels}, validation_split=VALIDATION_SPLIT, nb_epoch=50, batch_size=1024, shuffle=True,class_weight=class_weight,callbacks=[early_stopping, model_checkpoint])
model.load_weights(bst_model_path)
bst_val_score=min(hist.history['val_loss'])
preds = model.predict([test_data_1, test_data_2, aux_test],batch_size=4096, verbose=1)
preds += model.predict([test_data_2, test_data_1, aux_test], batch_size=4096, verbose=1)
preds /= 2.0
print(preds.shape)

out_df = pd.DataFrame({"test_id":test_labels, "is_duplicate":preds.ravel()})
out_df.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)
del out_df