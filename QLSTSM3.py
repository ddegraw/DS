# -*- coding: utf-8 -*-
import os
import csv
import codecs
import numpy as np
import pandas as pd
#np.random.seed(1337)
import gc
import re
import h5py

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
#from string import punctuation

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, merge, LSTM, Dropout, Bidirectional, Embedding, Lambda, Conv1D, MaxPooling1D
#from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten, Lambda
from keras.models import Model
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
#import sys

#sys.setdefaultencoding('utf-8')
re_weight = True
BASE_DIR = './input/'
GLOVE_DIR = 'D:/Embeddings/Glove/'
LEX_DIR = 'D:/Embeddings/LexVec/'
FAST_DIR = 'D:/Embeddings/FastText/'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
#MAX_SEQUENCE_LENGTH = 230
MAX_NB_WORDS = 220000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
#Some statistical analysis on the length of tokenized sequences
#mean = 11, std=6, 75%ile = 13, 95%ile = 23, 99%ile = 31, max = 245
MAX_SEQUENCE_LENGTH = 32
nb_words = 111375

'''
print('Indexing word vectors.')
embeddings_index = {}

#f = codecs.open(os.path.join(LEX_DIR, 'lexvec.300d.W.pos.vectors'), encoding='utf_8')
f = codecs.open(os.path.join(GLOVE_DIR, 'glove.840B.300d.txt'), encoding='utf_8')
#f = codecs.open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), encoding='utf_8')
for line in f:
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

def text_to_wordlist(text, remove_stopwords=True, stem_words=False, lemmatize = True):
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case and split them       

    #try: 
    #    text = text.lower()
    #except AttributeError: 
    #    pass
    # Optionally, remove stop words
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
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
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
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
    
    if lemmatize:
        text = text.split()
        lemmer = WordNetLemmatizer()
        lemmed_words = [lemmer.lemmatize(word) for word in text]
        text = " ".join(lemmed_words)
        
    # Return a list of words
    return(text)
    
texts_1 = [] 
texts_2 = []
labels = []

print('Processing text dataset')
#with codecs.open(TRAIN_DATA_FILE, encoding='utf_8',errors="ignore") as f:
#Original datafile needs to the decoded first, but excel csv does not
with codecs.open(TRAIN_DATA_FILE) as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts_1.append(text_to_wordlist(values[3]))
        texts_2.append(text_to_wordlist(values[4]))
        labels.append(int(values[5]))
print('Found %s training texts.' % len(texts_1))


test_texts_1 = []
test_texts_2 = []
test_labels = []  # list of label ids

#with codecs.open(TEST_DATA_FILE, encoding='utf_8',errors="ignore") as f:
with codecs.open(TEST_DATA_FILE) as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        test_texts_1.append(text_to_wordlist(values[1]))
        test_texts_2.append(text_to_wordlist(values[2]))
        test_labels.append(values[0])
print('Found %s test texts.' % len(test_texts_1))

texts_1 = [s.encode('utf_8') for s in texts_1]
texts_2 = [s.encode('utf_8') for s in texts_2]
test_texts_1 = [s.encode('utf_8') for s in test_texts_1]
test_texts_2 = [s.encode('utf_8') for s in test_texts_2]
test_labels = [s.encode('utf_8') for s in test_labels]

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
'''

if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None

# Model Architecture #
#num_lstm = np.random.randint(int(MAX_SEQUENCE_LENGTH*0.75), int(MAX_SEQUENCE_LENGTH*1.25))
num_lstm = 256
num_dense = 128
#num_dense = np.random.randint(int(0.75*num_lstm/2.0), int(1.25*num_lstm/2.0))
#rate_drop_lstm = 0.15 + np.random.rand() * 0.15
#rate_drop_dense = 0.15 + np.random.rand() * 0.15

rate_drop_lstm = 0.2
rate_drop_dense = 0.3

filts = 128
kern = 2
ps = 2

#rate_drop_lstm = 0.00
#rate_drop_dense = 0.00

act = 'relu'
re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'Conv2(%d,%d,%d)lstm_+Glove840B%d_%d_%.2f_%.2f'%(filts,kern, ps, num_lstm, num_dense, rate_drop_lstm, rate_drop_dense)

embedding_layer = Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
                            
#shared_lstm = Bidirectional(LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))
#shared_lstm = Bidirectional(LSTM(num_lstm))
shared_lstm = LSTM(num_lstm)


sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
es1 = embedding_layer(sequence_1_input)
x1 = Conv1D(filters=filts, kernel_size=3, padding='same', activation='relu')(es1)
x1 = MaxPooling1D(pool_size=ps)(x1)
x1 = Dropout(rate_drop_lstm)(x1)
x1 = Conv1D(filters=filts, kernel_size=kern, padding='same', activation='relu')(x1)
x1 = MaxPooling1D(pool_size=ps)(x1)
x1 = Dropout(rate_drop_lstm)(x1)
x1 = shared_lstm(x1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
es2 = embedding_layer(sequence_2_input)
y1 = Conv1D(filters=filts, kernel_size=3, padding='same', activation='relu')(es2)
y1 = MaxPooling1D(pool_size=ps)(y1)
y1 = Dropout(rate_drop_lstm)(y1)
y1 = Conv1D(filters=filts, kernel_size=kern, padding='same', activation='relu')(y1)
y1 = MaxPooling1D(pool_size=ps)(y1)
y1 = Dropout(rate_drop_lstm)(y1)
y1 = shared_lstm(y1)
#y1 = shared_lstm(y1)

merged = merge([x1,y1], mode='concat')
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)
merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)
merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)
merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)
preds = Dense(1, activation='sigmoid')(merged)

model = Model(input=[sequence_1_input,sequence_2_input], output=preds)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

print(STAMP)

early_stopping =EarlyStopping(monitor='val_loss', patience=4)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([data_1_train,data_2_train], labels_train, validation_data=([data_1_val, data_2_val], labels_val, weight_val), epochs=50, batch_size=256, shuffle=True,class_weight=class_weight,callbacks=[early_stopping, model_checkpoint])
  
model.load_weights(bst_model_path)
bst_val_score=min(hist.history['val_loss'])

preds = model.predict([test_data_1, test_data_2],batch_size=2048, verbose=1)
preds += model.predict([test_data_2, test_data_1], batch_size=2048, verbose=1)
preds /= 2.0
print(bst_val_score)

out_df = pd.DataFrame({"test_id":test_labels, "is_duplicate":preds.ravel()})
out_df.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)
del out_df
