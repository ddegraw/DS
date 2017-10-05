import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#from sklearn.model_selection import TimeSeriesSplit

from keras.models import Sequential
from keras.layers import Dense, Activation,Embedding
from keras.layers import LSTM, Dropout
from keras.optimizers import SGD, RMSprop, Adam

#from pandas.tseries.offsets import *
from keras import callbacks

remote = callbacks.RemoteMonitor(root='http://localhost:9000')

"""
with pd.HDFStore("C:\\Users\\4126694\\2sig-kaggle\\input\\train.h5", "r") as train:
    # Note that the "train" dataframe is the only dataframe in the file
    df = train.get("train")
   
ids = df["id"].unique()
ids_in = {}

for x in ids:
    time = df[df["id"] == x].timestamp
    if time.min() > 100 and time.max() < 1812:
        ids_in[x] = (time.min(), time.max())
        
""" 
def _load_data(data, n_prev = 61):  
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.05, n_prev = 61):  
    """
    This just splits data to training and testing parts
    """   
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = _load_data(df.iloc[0:ntrn], n_prev)
    X_test, y_test = _load_data(df.iloc[ntrn:], n_prev)

    return (X_train, y_train), (X_test, y_test)
    
hidden_neurons = 32
sequence_length = 1
bs = 5
#xtrain, xval = train_test_split(dfj)
#ytrain, yval = train_test_split(target)
#tstrain, tsval = train_test_split(ts,batch) 

model={}
predicted={}
#score={}

#for i in ids:
for i = 52:
    print i
    dfi = df[df["id"] == i]

    pd.set_option('mode.chained_assignment',None)
    dfi.loc[:,"cumprod"] = (1+dfi["y"]).cumprod()

    cols = [x for x in dfi.columns.values if x not in ["id", "timestamp","y","cumprod"]]
    l = len(cols)
    mean_values = dfi.mean(axis=0) 
    dfj = dfi.fillna(mean_values)
    #target = pd.DataFrame(dfj.pop('y'))
    target = pd.DataFrame(dfj.pop('cumprod'))
    ts = pd.DataFrame(dfj.pop('timestamp'))
    #dfj = dfi.drop(["id","y","cumprod"],axis=1)
    #dfj=dfj.fillna(0)
    #features = dfj.values    
    
    
    (X_train, y_train), (X_test, y_test) = train_test_split(target, n_prev = sequence_length) 

    model[i] = Sequential()
    model[i].add(LSTM(hidden_neurons, batch_input_shape=(None, sequence_length, 1), return_sequences=False))
    model[i].add(Dropout(0.1))
    model[i].add(Dense(1))
    model[i].add(Activation('linear'))
    #model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons), return_sequences=False))
    #model.add(LSTM(hidden_neurons, input_dim=length_of_sequences, return_sequences=True))

    #sgd = SGD(lr=0.005, decay=1e-6, momentum=0.1, nesterov=True)
    #rms = RMSprop(lr=0.005, rho=0.9, epsilon=1e-08, decay=0.1)
    #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)

    model[i].compile(optimizer='adam', loss='mse')
    model[i].fit(X_train, y_train, batch_size=bs, nb_epoch=10, validation_split=0.2)  
     
    #model.fit(X_train, y_train, batch_size=bs, nb_epoch=15, validation_data=(X_test, y_test), callbacks=[remote])     
     
    #predicted[instrument] = model.predict(X_test) 
    #dataf =  pd.DataFrame(predicted[:])
    #dataf.columns = ["predict"]
    #dataf["input"] = y_test[:]
    #dataf.plot(figsize=(15, 5))

    #fit = model.predict(X_train)
    #datag = pd.DataFrame(fit[:])
    #datag.columns = ["fit"]
    #datag["train"] = y_train[:]
    #datag.plot(figsize=(15, 5))

    #score = model.evaluate(X_test.as_matrix(), y_test, batch_size=16)
    #score[i] = model[i].evaluate(X_test, y_test, batch_size=16)
    