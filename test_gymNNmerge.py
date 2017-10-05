import kagglegym

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Merge

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Get the train dataframe
train = observation.train

ytrain = train['y'].values

mean_values = train.mean(axis=0)
train.fillna(mean_values, inplace=True)
standev = train.std(axis=0)
train = (train-mean_values)/(standev)

cols = [x for x in train.columns.values if x not in ["id","timestamp","y"]]


xtrain = train.drop(['id', 'y', 'timestamp'], axis=1)

#Split data into categorical data and rest
cat_tech = [0,2,6,9,10,11,12,14,16,17,18,22,29,32,34,37,38,39,42,43]
cat_cols = ["technical_" + str(s) for s in cat_tech]

#cat_fund = [9,10,11,17,19,21,22,26,27,33,34,35,41,42,48,49,62,63]
#cat_cols2 = ["fundamental_" + str(s) for s in cat_fund]

#cat_cols = cat_cols + cat_cols2

rest = [item for item in cols if item not in cat_cols]

l = len(cols)
lcat = len(cat_cols)

#split the columns
cat_data = xtrain[cat_cols]
xtrain = xtrain[rest]

del train

logb = Sequential()
logb.add(Dense(64, input_dim=lcat))
logb.add(Activation('sigmoid'))
logb.add(Dropout(0.1))
logb.add(Dense(32))
logb.add(Activation('sigmoid'))
logb.add(Dense(8))
logb.add(Activation('sigmoid'))

linb = Sequential()
linb.add(Dense(256, input_dim=l-lcat))
linb.add(Activation('linear'))
linb.add(Dropout(0.2))
linb.add(Dense(512))
linb.add(Activation('relu'))
linb.add(Dropout(0.2))
linb.add(Dense(64))
linb.add(Activation('linear'))
linb.add(Dense(8))
linb.add(Activation('linear'))

merged = Merge([logb, linb], mode='mul')

final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(1, activation='linear'))

final_model.compile(optimizer='Nadam', loss='mse')
print("Fitting the model")
final_model.fit([cat_data.as_matrix(), xtrain.as_matrix()], ytrain, batch_size=2000, nb_epoch=3, validation_split=0.2)

print('Running on the test set')
while True:
    #print('Running for test.')
    target = observation.target
    means = observation.features.mean(axis=0)
    test = observation.features.fillna(means)
    standev = test.std(axis=0)
    test = (test-mean_values)/(standev)       
    
    test = test.drop(['id','timestamp'], axis=1)
    
    test_cat = test[cat_cols]
    xtest = test[rest]
    
    #test_x = np.array(observation.features[col].values).reshape(-1,1)
    target.loc[:,'y'] = final_model.predict([test_cat.as_matrix(), xtest.as_matrix()])
    #observation.target.fillna(0, inplace=True)
    
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        print reward

    observation, reward, done, info = env.step(target)
    if done:
        break
    
print(info)