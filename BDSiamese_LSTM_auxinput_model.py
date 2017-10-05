# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import keras
#np.random.seed(1337)
#from string import punctuation

#from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, merge, LSTM, Dropout, Bidirectional, Embedding, Activation, Flatten
#from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten, Lambda
from keras.models import Model,Sequential
from keras.layers.wrappers import TimeDistributed
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
MAX_NB_WORDS = 111500
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
nb_words = 111375
#Some statistical analysis on the length of tokenized sequences
#mean = 11, std=6, 75%ile = 13, 95%ile = 23, 99%ile = 31, max = 245
MAX_SEQUENCE_LENGTH = 32

if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None

# Model Architecture #
def model():
    #num_lstm = np.random.randint(int(MAX_SEQUENCE_LENGTH*0.75), int(MAX_SEQUENCE_LENGTH*1.25))
    num_lstm = 32
    num_dense = 128
    #num_dense = np.random.randint(int(0.75*num_lstm/2.0), int(1.25*num_lstm/2.0))
    rate_drop_lstm = 0.15 + np.random.rand() * 0.10
    rate_drop_dense = 0.15 + np.random.rand() * 0.10

    act = 'relu'
    re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

    STAMP = 'lstm_aux_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, rate_drop_dense)

    embedding_layer = Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
                            
    #shared_lstm = Bidirectional(LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))
    #shared_lstm = Bidirectional(LSTM(num_lstm))
    shared_lstm = LSTM(num_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32',name='input1')
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
    #aux = Activation(activation='linear')(auxiliary_input)
    #aux = BatchNormalization()(aux)
    #merged = Dense(num_dense, activation=act)(merged_lstm)

    merged1 = merge([merged_lstm,auxiliary_input],mode='concat')
    #merged1 = keras.layers.concatenate([merged_lstm,auxiliary_input])

    merged1 = BatchNormalization()(merged1)
    merged1 = Dropout(rate_drop_dense)(merged1)
    merged1 = Dense(num_dense, activation=act)(merged1)
    merged1 = Dropout(rate_drop_dense)(merged1)
    merged1 = Dense(num_dense, activation=act)(merged1)
    merged1 = Dropout(rate_drop_dense)(merged1)
    merged1 = BatchNormalization()(merged1)
    predi = Dense(1, activation='sigmoid',name='main_output')(merged1)

    my_model = Model(input=[sequence_1_input,sequence_2_input,auxiliary_input], output=[predi,auxiliary_output])
    
    return STAMP, my_model
    
if __name__ == '__main__':

    STAMP, m = model()
    m.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'],loss_weights=[0.2, 1.0])
    

    early_stopping =EarlyStopping(monitor='val_loss', patience=3)
    bst_model_path = STAMP + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    #hist = m.fit({'input1': data_1_train, 'input2': data_2_train, 'aux_input': aux_train.as_matrix()}, {'main_output':labels_train, 'aux_output':labels_train}, validation_data=([data_1_val, data_2_val], labels_val, weight_val), nb_epoch=2, batch_size=256, shuffle=True,class_weight=class_weight,callbacks=[early_stopping, model_checkpoint])
    print(STAMP)
    train_mat = np.array(aux_train)
    
    hist = m.fit([data_1_train, data_2_train, train_mat], [labels_train, labels_train], validation_data=([data_1_val, data_2_val, aux_val], [labels_val,labels_val], [weight_val,weight_val]), epochs=2, batch_size=256, shuffle=True,class_weight=class_weight,callbacks=[early_stopping, model_checkpoint])

    model.load_weights(bst_model_path)
    bst_val_score=min(hist.history['val_loss'])
    preds = model.predict([test_data_1, test_data_2, aux_test],batch_size=2048, verbose=1)
    preds += model.predict([test_data_2, test_data_1, aux_test], batch_size=2048, verbose=1)
    preds /= 2.0
    print(preds.shape)

    out_df = pd.DataFrame({"test_id":test_labels, "is_duplicate":preds.ravel()})
    out_df.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)
    del out_df