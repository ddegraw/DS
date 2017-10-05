
import numpy as np
from keras.layers import merge, Input
from keras.layers.core import Dropout, Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import Model

MAX_SEQUENCE_LENGTH = 5
AUX_TRAIN_SHAPE = 64


# https://keras.io/getting-started/functional-api-guide/
def model():
    act = Activation('relu')
    output_num_lstm = 10
    num_dense = 10

    rate_drop_dense = 0.5

    # SHARED LAYERS: EMBEDDING + LSTM
    shared_embedding_layer = Embedding(input_dim=100, output_dim=64, input_length=MAX_SEQUENCE_LENGTH)
    shared_lstm = LSTM(output_num_lstm)

    # FIRST INPUT
    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='input1')
    embedded_sequences_1 = shared_embedding_layer(sequence_1_input)
    x1 = shared_lstm(embedded_sequences_1)

    # SECOND INPUT
    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='input2')
    embedded_sequences_2 = shared_embedding_layer(sequence_2_input)
    y1 = shared_lstm(embedded_sequences_2)

    # MERGE LSTM OUTPUT OF INPUT1 + INPUT2. OUTPUT.SHAPE = (?, num_dense)
    merged_lstm = merge([x1, y1], mode='concat')
    merged_lstm = Dropout(rate_drop_dense)(merged_lstm)
    merged_lstm = BatchNormalization()(merged_lstm)
    
    merged = Dense(num_dense, activation=act)(merged_lstm)

    # FLOW OUTPUT OF INPUT1 + INPUT2
    auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(merged_lstm)

    # THIRD INPUT. OUTPUT.SHAPE = (?, 32)
    auxiliary_input = Input(shape=(AUX_TRAIN_SHAPE,), name='aux_input')
    aux = Dense(32, activation=act)(auxiliary_input)

    # CONCATENATION OF LSTM RESULTS + AUX_INPUT, OUTPUT.SHAPE = (?, num_dense + 32).
    merged1 = merge([merged, aux], mode='concat')

    main_output = Dense(1, activation='sigmoid',name='main_output')(merged1)

    my_model = Model(inputs=[sequence_1_input,
                             sequence_2_input,
                             auxiliary_input],
                     outputs=[main_output,
                              auxiliary_output])
    return my_model


if __name__ == '__main__':
    num_samples = 1000
    inputs_1 = np.random.randint(low=0, high=100, size=(num_samples, MAX_SEQUENCE_LENGTH))
    inputs_2 = np.random.randint(low=0, high=100, size=(num_samples, MAX_SEQUENCE_LENGTH))
    aux_inputs = np.random.uniform(size=(num_samples, AUX_TRAIN_SHAPE))

    labels_1 = np.random.randint(low=0, high=2, size=(num_samples, 1))
    labels_2 = np.random.randint(low=0, high=2, size=(num_samples, 1))

    m = model()
    m.compile(optimizer='adam', loss='binary_crossentropy',
              loss_weights=[1., 0.2])

    m.fit([inputs_1, inputs_2, aux_inputs], [labels_1, labels_2],
          epochs=2, batch_size=32)

