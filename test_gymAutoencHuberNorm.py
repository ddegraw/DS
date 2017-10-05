import kagglegym
import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Input
from sklearn import linear_model

def train_test_split(data, test_size=0.2):  
    """
    This just splits data to training and validation parts
    """   
    #df = pd.DataFrame(data)    
    ntrn = round(len(data) * (1 - test_size))
    ntrn = int(ntrn)
    tt = data.iloc[0:ntrn]
    vv = data.iloc[ntrn:]
    ttrain = np.array(tt)
    vval = np.array(vv)
    return (ttrain, vval)     
       
def psi(u):
    a = np.percentile(u,70,axis=0)
    b = np.percentile(u,85,axis=0)
    c = np.percentile(u,95,axis=0)
    #v = pd.DataFrame().reindex_like(u)    
    v=pd.DataFrame()
    
    for i in range(train.columns.shape[0]):  
       
       foo1 = np.abs(u.ix[:,i]) >= 0
       bar1 = np.abs(u.ix[:,i]) < a[i]
       fubar1 = foo1 & bar1
          
       foo2 = np.abs(u.ix[:,i]) >= a[i]
       bar2 = np.abs(u.ix[:,i]) < b[i]
       fubar2 = foo2 & bar2
   
       foo3 = np.abs(u.ix[:,i]) >= b[i]
       bar3 = np.abs(u.ix[:,i]) < c[i]
       fubar3 = foo3 & bar3

       fubar4 = np.abs(u.ix[:,i]) >= c[i]  
       
       v[u.columns[i]] = u.ix[:,i]*fubar1 + a[i]*np.sign(u.ix[:,i])*fubar2 + a[i]*np.sign(u.ix[:,i])*(c[i]-np.abs(u.ix[:,i]))/(c[i]-b[i])*fubar3 + 0*u.ix[:,i]*fubar4  
               
    return v.mean(axis=0),v.std(axis=0)

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Get the train dataframe

train = observation.train
mean_values = train.mean(axis=0)
train.fillna(mean_values, inplace = True)
standev = train.std(axis=0)
#train = (train-mean_values)/(standev)
train = train.drop(['id','y', 'timestamp','fundamental_61','derived_1','fundamental_17','derived_4','fundamental_26','fundamental_23','fundamental_1'], axis=1)

'''Normalization'''
mean,stand = psi(train)
mean.fillna(mean_values, inplace = True)
stand.fillna(standev, inplace = True)
train = 0.5*(np.tanh(0.01*((train-mean)/stand))+1)

y = observation.train.y
(xtrain, xval) = train_test_split(train)
(ytrain, yval) = train_test_split(y) 

l = xtrain.shape[1]
encoding_dim = 45  

# this is our input placeholder
input_img = Input(shape=(l,))
# "encoded" is the encoded representation of the input
encoded = Dense(256, activation='linear')(input_img)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='linear')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(l, activation='linear')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)

# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model(for single layer model)
#decoder_layer = autoencoder.layers[-1]
# retrieve the last layer of the autoencoder model(for multi layer model)
decoder_layer1 = autoencoder.layers[-3]
decoder_layer2 = autoencoder.layers[-2]
decoder_layer3 = autoencoder.layers[-1]

# create the decoder model (Single layer model)
#decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
# create the decoder model(Multi-layer model)
decoder = Model(input=encoded_input, output=decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))
autoencoder.compile(optimizer='Nadam', loss='mse')

autoencoder.fit(xtrain, xtrain, nb_epoch=1, batch_size=2000, validation_data=(xval, xval),verbose=False)

reg = linear_model.LassoCV(max_iter=20000)
reg.fit(encoder.predict(xtrain),ytrain)

del xtrain
del ytrain
del xval
del yval

#mean_values = train.median(axis=0)
#standev = train.std(axis=0)

print('Running the test set')
while True:
    #print('Running for test.')
    #xtest = encoder.predict(test.as_matrix())
    #test_x = np.array(observation.features[col].values).reshape(-1,1)
    test = observation.features
    test = test.fillna(mean_values, inplace = True)
    test = test.drop(['id', 'timestamp','fundamental_61','derived_1','fundamental_17','derived_4','fundamental_26','fundamental_23','fundamental_1'], axis=1)
    standev = test.std(axis=0)
    #test = (test-mean_values)/(standev)
    mean,stand = psi(test)
    mean.fillna(mean_values, inplace = True)
    stand.fillna(standev, inplace = True)
    test = 0.5*(np.tanh(0.01*((test-mean)/stand))+1)    
    
    observation.target.loc[:,'y'] = reg.predict(encoder.predict(test.as_matrix()))
    target = observation.target
    #target.loc[:,'y'] = reg.predict(encoder.predict(test.as_matrix()))
    #observation.target.fillna(0, inplace=True)
    #observation.target.fillna(0, inplace=True)
    
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        print (reward)

    observation, reward, done, info = env.step(target)
    if done:
        break
    
print(info)