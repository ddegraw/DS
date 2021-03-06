import kagglegym
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

target = 'y'

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Get the train dataframe
train = observation.train
mean_values = train.mean(axis=0)
train.fillna(mean_values, inplace=True)
standev = train.std(axis=0)
train = (train-mean_values)/(standev)


y_train = train['y'].values
x_train = train.drop(['id', 'y', 'timestamp'], axis=1)

cols = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']

x_train=x_train[cols]

del train

l = x_train.shape[1]

model = Sequential()
model.add(Dense(128, input_dim=l))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(4))
model.add(Activation('linear'))
model.add(Dense(1))  
Activation('linear')

model.compile(optimizer='adam', loss='mse')
model.fit(x_train.as_matrix(), y_train, batch_size=2000, nb_epoch=5, verbose = 0, validation_split=0.2) 


print('Running for test.')
while True:
    
    target = observation.target
    test = observation.features
    mean_values = test.mean(axis=0)
    standev = test.std(axis=0)
    
    x_test = test.fillna(mean_values, inplace=True)
    x_test = (x_test-mean_values)/(standev)
    x_test = x_test.drop(['id', 'timestamp'], axis=1)
    
    x_test = x_test[cols]
    
    #test_x = np.array(observation.features[col].values).reshape(-1,1)
    target.loc[:,'y'] = model.predict(x_test.as_matrix())
    #observation.target.fillna(0, inplace=True)
    
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    observation, reward, done, info = env.step(target)
    if done:
        break
    
print(info)