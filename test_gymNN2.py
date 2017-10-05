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
final_model.fit([log_data, lin_data], targets)  # we pass one data array per model input



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
model.fit(x_train.as_matrix(), y_train, batch_size=2000, nb_epoch=30, validation_split=0.2) 

print('Running for test.')
while True:
    #print('Running for test.')
    target = observation.target
    means = observation.features.mean(axis=0)
    test = observation.features.fillna(means)
    
    test_x = test.drop(['id','timestamp'], axis=1)
    #test_x = np.array(observation.features[col].values).reshape(-1,1)
    target.loc[:,'y'] = model.predict(test_x.as_matrix())
    #observation.target.fillna(0, inplace=True)
    
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    observation, reward, done, info = env.step(target)
    if done:
        break
    
print(info)