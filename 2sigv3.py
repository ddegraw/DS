import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

from keras.models import Sequential
from keras.layers import Dense, Activation,TimeDistributedDense
from keras.layers import LSTM, Dropout
from keras.optimizers import SGD, RMSprop, Adam

from pandas.tseries.offsets import *
from keras import callbacks

remote = callbacks.RemoteMonitor(root='http://localhost:9000')

with pd.HDFStore("C:\\Users\\4126694\\2sig-kaggle\\input\\train.h5", "r") as train:
    # Note that the "train" dataframe is the only dataframe in the file
    df = train.get("train")
    
ids = df["id"].unique()
ids_in = {}
for x in ids:
    time = df[df["id"] == x].timestamp
    if time.min() > 100 and time.max() < 1812:
        ids_in[x] = (time.min(), time.max())

#instrument = 52
#dfi = df[df["id"] == instrument]
pd.set_option('mode.chained_assignment',None)
df.loc[:,"cumprod"] = (1+df["y"]).cumprod()

cols = [x for x in df.columns.values if x not in ["timestamp","y"]]
l = len(cols)

df = df.fillna(mean_values)
target = df.pop('y')
ts = df.pop('timestamp')
#features = df.values

def train_test_split(data, test_size=0.1):  
    """
    This just splits data to training and testing parts
    """   
    #df = pd.DataFrame(data)    
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    tt = data.iloc[0:ntrn]
    vv = data.iloc[ntrn:]
    train = np.array(tt)
    val = np.array(vv)
    return (train, val)

(xtrain, xval) = train_test_split(df)
(ytrain, yval) = train_test_split(target) 
(tstrain, tsval) = train_test_split(ts) 

model = Sequential()
model.add(Dense(256, input_dim=l))
model.add(Activation('linear'))
model.add(Dense(512))
model.add(Activation('linear'))
model.add(Dropout(0.25))
model.add(Dense(512))
model.add(Activation('linear'))
model.add(Dropout(0.1))
model.add(Dense(256))
model.add(Activation('linear'))
model.add(Dropout(0.1))
model.add(Dense(64))
model.add(Activation('linear'))
model.add(Dense(16))
model.add(Activation('linear'))
model.add(Dense(1))  
Activation('linear')

#sgd = SGD(lr=0.005, decay=1e-6, momentum=0.1, nesterov=True)
#rms = RMSprop(lr=0.005, rho=0.9, epsilon=1e-08, decay=0.1)
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)

model.compile(optimizer='adam', loss='mse')
model.fit(xtrain, ytrain, batch_size=10, nb_epoch=15, validation_split=0.2, callbacks=[remote])  

y_train = train['y'].values
x_train = train.drop(['id', 'y', 'timestamp'], axis=1)

del train

l = x_train.shape[1]
#print l

log_branch = Sequential()
log_branch.add(Dense(32, input_dim=784))

lin_branch = Sequential()
lin_branch.add(Dense(32, input_dim=784))

merged = Merge([log_branch, lin_branch], mode='concat')

final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(10, activation='softmax'))

final_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
final_model.fit([log_data_1, lin_data_2], targets)



predicted = model.predict(xtrain) 
dataf =  pd.DataFrame(predicted)
dataf.columns = ["predict"]
dataf["input"] = ytrain[:]
dataf.plot(figsize=(15, 5))

#score = model.evaluate(X_test.as_matrix(), y_test, batch_size=16)
score = model.evaluate(xval, yval, batch_size=16)