# -*- coding: utf-8 -*-
import os
import csv
import codecs
import numpy as np
import pandas as pd
np.random.seed(1337)
import gc

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, merge, LSTM, Lambda, Dropout, Bidirectional
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model, Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import sys

BASE_DIR = './input/'
GLOVE_DIR = 'D:/Embeddings/Glove/'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
'''

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
texts_1 = [] 
texts_2 = []
labels = []  # list of label ids
#with codecs.open(TRAIN_DATA_FILE, encoding='utf_8',errors="ignore") as f:
with codecs.open(TRAIN_DATA_FILE) as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts_1.append(values[3].decode('utf-8'))
        texts_2.append(values[4].decode('utf-8'))
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
        test_texts_1.append(values[1].decode('utf-8'))
        test_texts_2.append(values[2].decode('utf-8'))
        test_labels.append(values[0].decode('utf-8'))
print('Found %s texts.' % len(test_texts_1))

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
'''

# Model Architecture #
shared_lstm = Bidirectional(LSTM(64))

modela = Sequential()
modela.add(Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False))
modela.add(Conv1D(128, 5, activation='relu'))
modela.add(MaxPooling1D(5))

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
encoded_1 = modela(sequence_1_input)
x1 = shared_lstm(encoded_1)

modelb = Sequential()
modelb.add(Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False))
modelb.add(Conv1D(128, 5, activation='relu'))
modelb.add(MaxPooling1D(5))

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
encoded_2 = modelb(sequence_2_input)
y1 = shared_lstm(encoded_2)
                           

merged = merge([x1,y1], mode='concat')
#merged = BatchNormalization()(merged)
#merged = Dense(64, activation='relu')(merged)
#merged = Dropout(0.2)(merged)
merged = BatchNormalization()(merged)
preds = Dense(1, activation='sigmoid')(merged)
model = Model(input=[sequence_1_input,sequence_2_input], output=preds)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])



model.fit([data_1,data_2], labels, validation_split=VALIDATION_SPLIT, nb_epoch=2, batch_size=64, shuffle=True)
preds = model.predict([test_data_1, test_data_2])
print(preds.shape)

out_df = pd.DataFrame({"test_id":test_labels, "is_duplicate":preds.ravel()})
out_df.to_csv("test_predictions.csv", index=False)
del out_df