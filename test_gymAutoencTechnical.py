import kagglegym
import numpy as np

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

X = observation.train
y = observation.train.y
mean_values = X.median(axis=0)
X.fillna(mean_values, inplace=True)
standev = X.std(axis=0)
X = (X-mean_values)/(standev)
X = X.drop(['id', 'y','timestamp'], axis=1)

allCols = env.reset().train.columns
cols = [c for c in allCols if 'technical' in c]

(xtrain, xval) = train_test_split(X[cols])
(ytrain, yval) = train_test_split(y) 

l = xtrain.shape[1]
encoding_dim = 8
del X
del y


# this is our input placeholder
input_img = Input(shape=(l,))
# "encoded" is the encoded representation of the input
encoded = Dense(64, activation='linear')(input_img)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='linear')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
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

autoencoder.fit(xtrain, xtrain, nb_epoch=100, batch_size=2000, validation_data=(xval, xval),verbose=1)

reg = linear_model.LassoCV(max_iter=20000)
reg.fit(encoder.predict(xtrain),ytrain)

del xtrain
del ytrain
del xval
del yval

print('Running the test set')
while True:
    #print('Running for test.')
    target = observation.target
    test = observation.features
    mean_values = test.median(axis=0)
    standev = test.std(axis=0)
    
    test = test.fillna(mean_values, inplace=True)
    test = (test-mean_values)/(standev)
    test = test.drop(['id', 'timestamp'], axis=1)
    test = test[cols]
    #xtest = encoder.predict(test.as_matrix())
    #test_x = np.array(observation.features[col].values).reshape(-1,1)
    target.loc[:,'y'] = reg.predict(encoder.predict(test.as_matrix()))
    #observation.target.fillna(0, inplace=True)
    del test
    #observation.target.fillna(0, inplace=True)
    
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        print (reward)

    observation, reward, done, info = env.step(target)
    if done:
        break
    
print(info)