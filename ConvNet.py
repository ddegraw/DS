# -*- coding: utf-8 -*-

import numpy as np
#np.random.seed(1337)
from keras.layers import Dense, Input, Flatten, merge, Dropout, Reshape
from keras.layers import Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D
from keras.models import Model
from keras.layers.normalization import BatchNormalization


BASE_DIR = './input/'
MAX_SEQUENCE_LENGTH = 32
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
nb_words = 111375

filter_sizes = (3, 8)
num_filters = 16
dropout_prob = (0.5, 0.8)
num_dense = 64

STAMP = 'CNN%d_%d_%.2f'%(num_filters, num_dense, dropout_prob[0])
# Model Architecture #
embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
embedded_sequences_1 = Reshape((300,32))(embedded_sequences_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
embedded_sequences_2 = Reshape((300,32))(embedded_sequences_2)

'''
x = Conv1D(16, 3, activation='relu', input_dim = 300, input_length = 32)(embedded_sequences_1)
x = MaxPooling1D(5,3)(x)
x = Conv1D(16, 3, activation='relu')(x)
x = MaxPooling1D(4,2)(x)
x = Conv1D(16, 3, activation='relu')(x)
x = MaxPooling1D(5,1)(x)  # global max pooling

#x = Reshape((16,41))(x)
x = Dense(128, activation='relu')(x)

y = Conv1D(16, 3, activation='relu', input_dim = 300, input_length = 32)(embedded_sequences_2)
y = MaxPooling1D(5,3)(y)
y = Conv1D(16, 3, activation='relu')(y)
y = MaxPooling1D(4,2)(y)
y = Conv1D(16, 3, activation='relu')(y)
y = MaxPooling1D(5,1)(y)  # global max pooling

#y = Reshape((16,41))(x)
y = Dense(128, activation='relu')(y)

merged = merge([x,y], mode='concat')

'''
convx =[]
for sz in filter_sizes:
    convx1 = Conv1D(16, sz, activation="relu")(embedded_sequences_1)
    convx1 = MaxPooling1D(5,2)(convx1)
    convx1 = Flatten()(convx1)
    convx.append(convx1)   
x1 = merge(convx, mode='concat')

convy =[]
for sz in filter_sizes:
    convy1 = Conv1D(16, sz, activation="relu")(embedded_sequences_2)
    convy1 = MaxPooling1D(5,2)(convy1)
    convy1 = Flatten()(convy1)
    convy.append(convy1)   
y1 = merge(convy, mode='concat')                   

merged = merge([x1,y1], mode='concat')
merged = Dropout(0.5)(merged)
merged = BatchNormalization()(merged)
merged = Dense(128, activation='relu')(merged)
merged = Dropout(0.2)(merged)
merged = Dense(32, activation='relu')(merged)
merged = Dropout(0.15)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation='sigmoid')(merged)
model = Model(input=[sequence_1_input,sequence_2_input], output=preds)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
print(model.summary())
