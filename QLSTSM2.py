# -*- coding: utf-8 -*-
import os
import csv
import codecs
import numpy as np
import pandas as pd
np.random.seed(1337)
import gc
from time import time

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, merge, LSTM, Lambda, Dropout, Bidirectional
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.callbacks import ModelCheckpoint

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
# Train params
NB_EPOCHS = 2
BATCH_SIZE = 1024
VAL_SPLIT = 0.1
WEIGHTS_PATH = 'lstm_weights.h5'
SUBMIT_PATH = 'lstm_submission_1.csv'

df_train = pd.read_csv('./input/train.csv')

texts = df_train[['question1','question2']]
labels = df_train['is_duplicate']

del df_train


#TOKENIZE TEXT
tk = Tokenizer(nb_words=MAX_NB_WORDS)

tk.fit_on_texts(list(texts.question1.values.astype(str)) + list(texts.question2.values.astype(str)))
x1 = tk.texts_to_sequences(texts.question1.values.astype(str))
x1 = pad_sequences(x1, maxlen=MAX_SEQUENCE_LENGTH)

x2 = tk.texts_to_sequences(texts.question2.values.astype(str))
x2 = pad_sequences(x2, maxlen=MAX_SEQUENCE_LENGTH)

# Preprocessing Test
print("Acquiring Test Data")
t0 = time()
df_test = pd.read_csv('./input/test.csv')
print("Done! Acquisition time:", time()-t0)

# Preprocessing
print("Preprocessing test data")
t0 = time()

i = 0
while True:
    if (i*BATCH_SIZE > df_test.shape[0]):
        break
    t1 = time()
    tk.fit_on_texts(list(df_test.iloc[i*BATCH_SIZE:(i+1)*BATCH_SIZE].question1.values.astype(str))
                    + list(df_test.iloc[i*BATCH_SIZE:(i+1)*BATCH_SIZE].question2.values.astype(str)))
    i += 1
    if (i % 100 == 0):
        print("Preprocessed Batch {0}/{1}, Word index size: {2}, ETC: {3} seconds".format(i,
                                                                int(df_test.shape[0]/BATCH_SIZE+1),
                                                                len(tk.word_index),
                                                                int(int(df_test.shape[0]/BATCH_SIZE+1)-i)*(time()-t1)))

word_index = tk.word_index

print("Done! Preprocessing time:", time()-t0)
print("Word index length:",len(word_index))

print('Shape of data tensor:', x1.shape, x2.shape)
print('Shape of label tensor:', labels.shape)

def get_model(p_drop=0.0):
    embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
                            
    shared_lstm = Bidirectional(LSTM(64))
    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = shared_lstm(embedded_sequences_1)


    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = shared_lstm(embedded_sequences_2)

    merged = merge([x1,y1], mode='concat')
    merged = BatchNormalization()(merged)
    preds = Dense(1, activation='sigmoid')(merged)
    model = Model(input=[sequence_1_input,sequence_2_input], output=preds)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    
    return model
    
   '''   
model = get_model(p_drop=0.2)
#checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc', save_best_only=True, verbose=2)

model.fit([x1, x2], y=labels, batch_size=1024, nb_epoch=2,
                 verbose=1, validation_split=0.1, shuffle=True, callbacks=[checkpoint])
            
# Predicting
i = 0
predictions = np.empty([df_test.shape[0],1])
while True:
    t1 = time()
    if (i * BATCH_SIZE > df_test.shape[0]):
        break
    x1 = pad_sequences(tk.texts_to_sequences(
        df_test.question1.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].values.astype(str)), maxlen=MAX_SEQUENCE_LENGTH)
    x2 = pad_sequences(tk.texts_to_sequences(
        df_test.question2.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].values.astype(str)), maxlen=MAX_SEQUENCE_LENGTH)
    try:
        predictions[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = model.predict([x1, x2], batch_size=BATCH_SIZE, verbose=0)
    except ValueError:
        predictions[i*BATCH_SIZE:] = model.predict([x1, x2], batch_size=BATCH_SIZE, verbose=0)[:(df_test.shape[0]-i*BATCH_SIZE)]

    i += 1
    if (i % 100 == 0):
        print("Predicted Batch {0}/{1}, ETC: {2} seconds".format(i,
                                                                int(df_test.shape[0]/BATCH_SIZE),
                                                                int(int(df_test.shape[0]/BATCH_SIZE+1)-i)*(time()-t1)))


df_test["is_duplicate"] = predictions


df_test[['test_id','is_duplicate']].to_csv(SUBMIT_PATH, header=True, index=False)
print("Done!")
print("Submission file saved to:",check_output(["ls", SUBMIT_PATH]).decode("utf8"))