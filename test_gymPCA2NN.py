import kagglegym
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn import decomposition


# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

low_y_cut = -0.086093
high_y_cut = 0.093497
y = observation.train.y
y_is_above_cut = (y > high_y_cut)
y_is_below_cut = (y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

# Get the train dataframe
train = observation.train
train = train.drop(['id','y', 'timestamp','fundamental_61','derived_1','fundamental_17','derived_4','fundamental_26','fundamental_23','fundamental_1'], axis=1)
mean_values = train.loc[y_is_within_cut, :].mean(axis=0)
train.fillna(mean_values, inplace = True)
standev = train.loc[y_is_within_cut, :].std(axis=0)
train = (train.loc[y_is_within_cut, :]-mean_values)/(standev)

#Transform data matrix
pca = decomposition.PCA(n_components=50)
pca.fit(train.as_matrix())
train = pca.fit_transform(train.as_matrix())
l = train.shape[1]

model = Sequential()
model.add(Dense(64, input_dim=l))
model.add(Activation('linear'))
model.add(Dense(128))
model.add(Activation('linear'))
model.add(Dropout(0.1))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(16))
model.add(Activation('linear'))
model.add(Dropout(0.1))
model.add(Dense(4))
model.add(Activation('linear'))
model.add(Dense(1))  
Activation('linear')

model.compile(optimizer='adam', loss='mse')
model.fit(train, y.loc[y_is_within_cut], batch_size=2000, nb_epoch=1, verbose = 1, validation_split=0.2) 

del train

print('Running for test.')
while True:
    
    target = observation.target
    test = observation.features
    test = test.drop(['id', 'timestamp','fundamental_61','derived_1','fundamental_17','derived_4','fundamental_26','fundamental_23','fundamental_1'], axis=1)
    mean_values = test.mean(axis=0)
    standev = test.std(axis=0)
    
    test = test.fillna(mean_values, inplace = True)
    test = (test-mean_values)/(standev)
    #test_x = np.array(observation.features[col].values).reshape(-1,1)
    target.loc[:,'y'] = model.predict(pca.fit_transform(test.as_matrix()))
    #observation.target.fillna(0, inplace=True)
    
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        print(reward)

    observation, reward, done, info = env.step(target)
    if done:
        break
    
print(info)