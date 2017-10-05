import kagglegym
import numpy as np
from scipy.stats import mode

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

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Get the train dataframe

train = observation.train

allCols = train.columns
allCols = allCols.drop(['id', 'y', 'timestamp'])
cols = [c for c in allCols if 'fundamental' in c] 
rest = [item for item in allCols if item not in cols]

mean_values_fund = train[cols].median(axis=0)
train[cols].fillna(mean_values_fund, inplace = True)
standev_fund = train[cols].std(axis=0)
train[cols] = (train[cols]-mean_values_fund)/(standev_fund)

mean_values_rest = train[rest].mean(axis=0)
train[rest].fillna(mean_values_rest, inplace = True)
standev_rest = train[rest].std(axis=0)
train[rest] = (train[rest]-mean_values_rest)/(standev_rest)


train = train.drop(['id','y', 'timestamp'], axis=1)
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

autoencoder.fit(xtrain, xtrain, nb_epoch=1, batch_size=2000, validation_data=(xval, xval),verbose=True)

reg = linear_model.LassoCV(max_iter=20000)
reg.fit(encoder.predict(xtrain),ytrain)

del xtrain
del ytrain
del xval
del yval

#means = train.median(axis=0)
#standev = train.std(axis=0)

print('Running the test set')
while True:
    #print('Running for test.')
    #xtest = encoder.predict(test.as_matrix())
    #test_x = np.array(observation.features[col].values).reshape(-1,1)
    test = observation.features
    
    test[cols].fillna(mean_values_fund, inplace = True)
    standev_fund = test[cols].std(axis=0)
    means = test[cols].mode(axis=0)
    test[cols] = (test[cols]-means)/(standev_fund)

    test[rest].fillna(mean_values_rest, inplace = True)
    standev_rest = test[rest].std(axis=0)
    means1 = test[rest].median(axis=0)
    test[rest] = (test[rest]-means1)/(standev_rest)   

    test = test.drop(['id', 'timestamp'], axis=1)
    
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