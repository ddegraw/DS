import kagglegym
import numpy as np
import pandas as pd
from scipy import stats

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Get the train dataframe
train = observation.train
#mean_values = train.mean(axis=0)
#standev = train.std(axis=0)

means = train.drop(['y', 'timestamp'], axis=1).groupby('id').agg([np.mean]).reset_index()
stds = train.drop(['y', 'timestamp'], axis=1).groupby('id').agg([np.std]).reset_index()

means.fillna(0)
stds.fillna(0)
sec = means['id']


y = np.array(train.y)

xtrain = train

for s in sec:
    temp = train[train["id"]== s]
    temp.fillna(means[s], inplace=True)
    xtrain[xtrain["id"]== s] = (temp-means[s])/(stds[s])
    
   
mean_values = train.mean(axis=0)
train.fillna(mean_values, inplace=True)

xtrain = x_train.drop(['id', 'y', 'timestamp'], axis=1)
del train

l = x_train.shape[1]
#print l

model = Sequential()
model.add(Dense(256, input_dim=l))
model.add(Activation('sigmoid'))
model.add(Dense(512))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(32))
model.add(Activation('linear'))
model.add(Dense(4))
model.add(Activation('linear'))
model.add(Dense(1))  
Activation('linear')

model.compile(optimizer='Nadam', loss='mse')
model.fit(xtrain.as_matrix(), y, batch_size=2000, nb_epoch=5, validation_split=0.2) 

print('Running for test.')
while True:
    #print('Running for test.')
    target = observation.target
    test = observation.features

    mean = test.drop(['timestamp'], axis=1).groupby('id').agg([np.mean]).reset_index()
    std = test.drop(['timestamp'], axis=1).groupby('id').agg([np.std]).reset_index()

    mean.fillna(0)
    std.fillna(0)
    sec = mean['id']

    xtest = test

    for s in sec:
        temp = test[test["id"]== s]
        temp.fillna(mean[s], inplace=True)
        xtest[xtest["id"] == s] = (temp-mean[s])/(std[s])
            
    #test_x = np.array(observation.features[col].values).reshape(-1,1)
    target.loc[:,'y'] = model.predict(xtest.as_matrix())
    #observation.target.fillna(0, inplace=True)
    
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    observation, reward, done, info = env.step(target)
    if done:
        break
    
print(info)