import kagglegym

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn import decomposition

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Get the train dataframe
train = observation.train

#mean_values = train.mean(axis=0)
#standev = train.std(axis=0)
mean_values = train.mean(axis=0)
#standev = train.std(axis=0)
train.fillna(mean_values, inplace=True)
#train = (train-mean_values)/(standev)


train = train.drop(['id', 'timestamp'], axis=1)

pca = decomposition.PCA(n_components=10)
#pipe = Pipeline(steps=[('pca', pca), ('regression', regression)])

#X = xtrain.loc[y_is_within_cut,:]
#Y = y.loc[y_is_within_cut]

pca.fit(train.as_matrix())
xtrain = pca.fit_transform(train.as_matrix())
l = xtrain.shape[1]
#print l


model = Sequential()
model.add(Dense(32, input_dim=l))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(12))
model.add(Activation('linear'))
model.add(Dense(6))
model.add(Activation('linear'))
model.add(Dense(1))  
Activation('linear')

model.compile(optimizer='Nadam', loss='mse')
model.fit(xtrain, y, batch_size=2000, nb_epoch=5, validation_split=0.25) 

print('Running the test set')
while True:
    #print('Running for test.')
    target = observation.target
    test = observation.features
 
    means = observation.features.mean(axis=0)
    #standev = test.std(axis=0)
    test = observation.features.fillna(means)
    #test = (test-mean_values)/(standev)       
    
    test = test.drop(['id', 'timestamp'], axis=1)
    xtest = pca.fit_transform(test.as_matrix())
    #test_x = np.array(observation.features[col].values).reshape(-1,1)
    target.loc[:,'y'] = model.predict(xtest)
    #observation.target.fillna(0, inplace=True)
    
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        print (reward)

    observation, reward, done, info = env.step(target)
    if done:
        break
    
print(info)