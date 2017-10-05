# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
#np.random.seed(1337)
#from string import punctuation

#from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, merge, LSTM, Dropout, Bidirectional, Embedding, Activation, Merge, Reshape, TimeDistributed
#from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten, Lambda
from keras.models import Model,Sequential
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
MAX_NB_WORDS = 110000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
#Some statistical analysis on the length of tokenized sequences
#mean = 11, std=6, 75%ile = 13, 95%ile = 23, 99%ile = 31, max = 245
MAX_SEQUENCE_LENGTH = 32


# Model Architecture #
#num_lstm = np.random.randint(int(MAX_SEQUENCE_LENGTH*0.75), int(MAX_SEQUENCE_LENGTH*1.25))
num_lstm = 32
num_dense = 64
#num_dense = np.random.randint(int(0.75*num_lstm/2.0), int(1.25*num_lstm/2.0))
rate_drop_lstm = 0.10 + np.random.rand() * 0.05
rate_drop_dense = 0.10 + np.random.rand() * 0.05

act = 'relu'
re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'lstm_aux_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, rate_drop_dense)

shared_lstm = LSTM(num_lstm)

x1 = Sequential()
x1.add(Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
x1.add(shared_lstm)
x1.build()

y1 = Sequential()
y1.add(Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
y1.add(shared_lstm)
y1.build()

merged_lstm = Merge([x1,y1],mode='concat')

z1 = Sequential()
z1.add(merged_lstm)
z1.add(Dropout(rate_drop_dense))
z1.add(BatchNormalization())
z1.build()

auxiliary_input = Sequential()
auxiliary_input.add(Dense(128,input_shape=(aux_train.shape[1],)))
auxiliary_input.add(Activation('relu'))
auxiliary_input.add(Dense(64))
auxiliary_input.add(Activation('relu'))
auxiliary_input.add(BatchNormalization())
auxiliary_input.build()
                  
merged = Merge([z1,auxiliary_input],mode='concat')

model=Sequential()
model.add(merged)
#model.add(Reshape((EMBEDDING_DIM,)))
model.add(BatchNormalization())
model.add(Dense(num_dense,activation=act))
model.add(Dropout(rate_drop_dense))
model.add(Dense(num_dense,activation=act))
model.add(Dropout(rate_drop_dense))
model.add(Dense(1,activation='sigmoid'))


#model = Model(input=[sequence_1_input,sequence_2_input,auxiliary_input], output=[preds,auxiliary_output])
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
print(STAMP)

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([ data_1_train, data_2_train, aux_train], labels_train, validation_data=([data_1_val, data_2_val, aux_val], labels_val, weight_val), nb_epoch=3, batch_size=512, shuffle=True,class_weight=class_weight,callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
bst_val_score=min(hist.history['val_loss'])
preds = model.predict([test_data_1, test_data_2, aux_test],batch_size=2048, verbose=1)
preds += model.predict([test_data_2, test_data_1, aux_test], batch_size=2018, verbose=1)
preds /= 2.0
print(preds.shape)

out_df = pd.DataFrame({"test_id":test_labels, "is_duplicate":preds.ravel()})
out_df.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)
del out_df