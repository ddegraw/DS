# -*- coding: utf-8 -*-

import numpy as np
#np.random.seed(1337)
from keras.layers import Dense, Input, Flatten, merge, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D
from keras.models import Model,Sequential

from keras.layers.normalization import BatchNormalization

BASE_DIR = './input/'
MAX_SEQUENCE_LENGTH = 32
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
num_dense = 64

STAMP = 'CNN%d_%d_%.2f'%(num_filters, num_dense, dropout_prob[0])
# Model Architecture #
embedding_layer = Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)

convx1 = Sequential()
convx1.add(Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
convx1.add(Conv1D(filters=num_filters, kernel_size=sz, padding="valid", activation="relu", strides=1))
convx1.add(GlobalMaxPooling1D(pool_size=2))
convx1.add(BatchNormalization())
convx1.add(Activation('relu'))
convx1.add(Dropout(.20))
convx1.add(Flatten())
convx1.add(LSTM(32))

convy1 = Sequential()
convy1.add(Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
convy1.add(Conv1D(filters=num_filters, kernel_size=sz, padding="valid", activation="relu", strides=1))
convy1.add(GlobalMaxPooling1D(pool_size=2))
convy1.add(BatchNormalization())
convy1.add(Activation('relu'))
convy1.add(Dropout(.20))
convy1.add(Flatten())
convy1.add(LSTM(32))                  







merged = merge([x1,y1], mode='concat')


model = Sequential()
model.add(merged)
'''
merged = Dropout(0.5)(merged)
merged = BatchNormalization()(merged)
merged = Dense(num_dense, activation='relu')(merged)
merged = Dropout(0.15)(merged)
merged = BatchNormalization()(merged)
'''
model.add(Dense(1, activation='sigmoid'))
#preds = Dense(1, activation='sigmoid')(merged)
#model = Model(input=[sequence_1_input,sequence_2_input], output=preds)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
