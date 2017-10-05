import kagglegym
import numpy as np
import pandas as pd

from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Dropout  
from keras.layers.recurrent import LSTM

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()
# Get the train dataframe

train = observation.train

low_y_cut = -0.075
high_y_cut = 0.075

y = train['y'].clip(low_y_cut, high_y_cut).copy()
d_mean= train.median(axis=0)

n = train.isnull().sum(axis=1)
for c in train.columns:
    train[c + '_nan_'] = pd.isnull(train[c])
    d_mean[c + '_nan_'] = 0
train = train.fillna(d_mean)
train['znull'] = n

l = train.shape[1]

def _load_data(data, n_prev = 61):  

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev])
        docY.append(data.iloc[i+n_prev])
    alsX = docX
    alsY = docY

    return alsX, alsY

def train_test_split(df, test_size=0.1, n_prev = 300):  

    #This just splits data to training and testing parts
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = _load_data(df.iloc[0:ntrn], n_prev)
    X_test, y_test = _load_data(df.iloc[ntrn:], n_prev)

    return (X_train, y_train), (X_test, y_test)
 
#(Xtrain1, ytrain1), (X_test, y_test) = train_test_split(train, n_prev = length_of_sequences)
(Xtrain2, ytrain2), (X_test, y_test) = train_test_split(y, n_prev = length_of_sequences)  


length_of_sequences = 300
in_out_neurons = 1
hidden_neurons = 20
bs = 2000

model = Sequential()  
model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons), return_sequences=False))
#model.add(LSTM(hidden_neurons, input_dim=length_of_sequences, return_sequences=True))
model.add(Dropout(0.1))
#model.add(TimeDistributedDense(length_of_sequences))
model.add(Dense(in_out_neurons))
model.add(Activation("linear"))  
model.compile(loss="mean_squared_error", optimizer="adam")

model.fit(Xtrain2, ytrain2, batch_size=bs, nb_epoch=15, validation_split=0.2)  
     
  
print('Running for test.')
while True:
    test = o.features
    n = test.isnull().sum(axis=1)
    for c in test.columns:
        test[c + '_nan_'] = pd.isnull(test[c])
    test = test.fillna(d_mean)
    test['znull'] = n
    
    pred = o.target
    test2 = test[['technical_30', 'technical_20', 'fundamental_11']].as_matrix()
    #test2 = np.array(test2.fillna(d_mean).values).reshape(-1,1)
    lamb = 0.4
    pred['y'] = (model1.predict(test[datacols]).clip(low_y_cut, high_y_cut) * lamb) + (model2.predict(test2).clip(low_y_cut, high_y_cut) * (1.0-lamb))
    pred['y'] =  model.predict(pred['y'].as_matrix())      
    pred['y'] = pred.apply(lambda r: 0.95 * r['y'] + 0.05 * ymean_dict[r['id']] if r['id'] in ymean_dict else r['y'], axis = 1)
    pred['y'] = [float(format(x, '.6f')) for x in pred['y']]
    o, reward, done, info = env.step(pred)
    if done:
        print("el fin ...", info["public_score"])
        break
    if o.features.timestamp[0] % 100 == 0:
        print(reward)
       
print(info)