
import numpy as np

#np.random.seed(1337)
#from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, merge, LSTM, Dropout, Bidirectional, Embedding, Lambda, Flatten, Reshape
#from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten, Lambda
from keras.models import Model
#from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from attention_lstm import AttentionLSTMWrapper
from keras.callbacks import EarlyStopping, ModelCheckpoint
#import sys
re_weight = True

if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None
#sys.setdefaultencoding('utf-8')
BASE_DIR = './input/'
VALIDATION_SPLIT = 0.1
#Some statistical analysis on the length of tokenized sequences
#mean = 11, std=6, 75%ile = 13, 95%ile = 23, 99%ile = 31, max = 245

# Model Architecture #
#num_lstm = np.random.randint(int(MAX_SEQUENCE_LENGTH*0.75), int(MAX_SEQUENCE_LENGTH*1.25))
num_lstm = 64
num_dense = 32
#num_dense = np.random.randint(int(0.75*num_lstm/2.0), int(1.25*num_lstm/2.0))
rate_drop_lstm = 0.15 + np.random.rand() * 0.1
rate_drop_dense = 0.15 + np.random.rand() * 0.1

act = 'relu'
re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'lstmAttn_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, rate_drop_dense)

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedding_layer = Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)

embedded_sequences_1 = embedding_layer(sequence_1_input)
embedded_sequences_2 = embedding_layer(sequence_2_input)

lstm1 = LSTM(num_lstm, return_sequences=True)
lstm2 = LSTM(num_lstm, return_sequences=True)

x1 = lstm1(embedded_sequences_1)
y1 = lstm2(embedded_sequences_2)

maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape = lambda x: (x[0],x[2]))
maxpool.supports_masking = True
x1 = maxpool(x1)
y1 = maxpool(y1)

lstma = AttentionLSTMWrapper(lstm1, y1, single_attention_param=True)
lstmb = AttentionLSTMWrapper(lstm2, x1, single_attention_param=True)

lstm1a = lstma(embedded_sequences_1)
lstm2b = lstmb(embedded_sequences_2)

#Do I need a maxpool layer here?
merged = merge([lstm1a,lstm2b], mode='concat')
#merged = Reshape((EMBEDDING_DIM,))(merged)
merged = Flatten()(merged)
merged = BatchNormalization()(merged)
merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = Dense(16, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
preds = Dense(1, activation='sigmoid')(merged)

model = Model(input=[sequence_1_input,sequence_2_input], output=preds)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

print(STAMP)
bst_model_path = STAMP + '.h5'
early_stopping =EarlyStopping(monitor='val_loss', patience=3)

model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([data_1,data_2], labels, validation_data=([data_1_val, data_2_val], labels_val, weight_val), nb_epoch=100, batch_size=256, shuffle=True ,class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])
  
model.load_weights(bst_model_path)
bst_val_score=min(hist.history['val_loss'])
preds = model.predict([test_data_1, test_data_2],batch_size=4096, verbose=1)
preds += model.predict([test_data_2, test_data_1], batch_size=4096, verbose=1)
preds /= 2.0
print(bst_val_score)

out_df = pd.DataFrame({"test_id":test_labels, "is_duplicate":preds.ravel()})
out_df.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)
del out_df
