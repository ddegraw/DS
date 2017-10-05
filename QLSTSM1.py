# -*- coding: utf-8 -*-
import os
import csv
import codecs
import numpy as np
import pandas as pd
np.random.seed(1337)
import gc
import re

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, merge, LSTM, Dropout, Bidirectional
#from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten, Lambda
from keras.models import Model
#from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras import backend as K
import sys

sys.setdefaultencoding('utf-8')

BASE_DIR = './input/'
GLOVE_DIR = 'D:/Embeddings/Glove/'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
#MAX_SEQUENCE_LENGTH = 230
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = np.random.randint(100, 150)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

act = 'relu'
re_weight = True # whether to re-weight classes to fit the 17.5% share in test set


STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, rate_drop_dense)


print('Indexing word vectors.')
embeddings_index = {}
f = codecs.open(os.path.join(GLOVE_DIR, 'glove.840B.300d.txt'), encoding='utf_8')
for line in f:
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

print('Processing text dataset')

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
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
    
texts_1 = [] 
texts_2 = []
labels = []

#with codecs.open(TRAIN_DATA_FILE, encoding='utf_8',errors="ignore") as f:
with codecs.open(TRAIN_DATA_FILE) as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts_1.append(text_to_wordlist(values[3].decode('utf-8')))
        texts_2.append(text_to_wordlist(values[4].decode('utf-8')))
        labels.append(int(values[5].decode('utf-8')))
print('Found %s texts.' % len(texts_1))

test_texts_1 = []
test_texts_2 = []
test_labels = []  # list of label ids
#with codecs.open(TEST_DATA_FILE, encoding='utf_8',errors="ignore") as f:
with codecs.open(TEST_DATA_FILE) as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        test_texts_1.append(text_to_wordlist(values[1].decode('utf-8')))
        test_texts_2.append(text_to_wordlist(values[2].decode('utf-8')))
        test_labels.append(values[0].decode('utf-8'))
print('Found %s texts.' % len(test_texts_1))

texts_1 = [s.encode('utf_8') for s in texts_1]
texts_2 = [s.encode('utf_8') for s in texts_2]
test_texts_1 = [s.encode('utf_8') for s in test_texts_1]
test_texts_2 = [s.encode('utf_8') for s in test_texts_2]
test_labels = [s.encode('utf_8') for s in test_labels]

MAX_SEQUENCE_LENGTH = max(len(texts_1 + texts_2 + test_texts_1 + test_texts_2))

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)
sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

#Need to think about removing non-english words

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

# Model Architecture #

embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
                            
shared_lstm = Bidirectional(LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))


sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = shared_lstm(embedded_sequences_1)

#x1 = Conv1D(128, 3, activation='relu')(embedded_sequences_1)
#x1 = MaxPooling1D(10)(x1)
#x1 = Flatten()(x1)
#x1 = Dense(64, activation='relu')(x1)
#x1 = Dropout(0.2)(x1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = shared_lstm(embedded_sequences_2)

#y1 = Conv1D(128, 3, activation='relu')(embedded_sequences_2)
#y1 = MaxPooling1D(10)(y1)
#y1 = Flatten()(y1)
#y1 = Dense(64, activation='relu')(y1)
#y1 = Dropout(0.2)(y1)

merged = merge([x1,y1], mode='concat')
#merged = BatchNormalization()(merged)
#merged = Dense(64, activation='relu')(merged)
#merged = Dropout(0.2)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)
merged = Dropout(rate_drop_dense)(merged)

merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation='sigmoid')(merged)


model = Model(input=[sequence_1_input,sequence_2_input], output=preds)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
print(STAMP)

early_stopping =EarlyStopping(monitor='val_loss', patience=1)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

model.fit([data_1,data_2], labels, validation_split=VALIDATION_SPLIT, nb_epoch=10, batch_size=256, shuffle=True,class_weight=class_weight,callbacks=[early_stopping, model_checkpoint])
preds = model.predict([test_data_1, test_data_2])
print(preds.shape)

out_df = pd.DataFrame({"test_id":test_labels, "is_duplicate":preds.ravel()})
out_df.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)
del out_df