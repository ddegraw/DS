import kagglegym
import numpy as np

from keras.models import Model
from keras.layers import Dense, Input
from sklearn import linear_model, ensemble

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
train = train.drop(['id','y', 'timestamp','derived_1','derived_3',\
'fundamental_6','fundamental_10','fundamental_12','fundamental_17',\
'fundamental_18','fundamental_19','fundamental_20','fundamental_23','fundamental_26',\
'fundamental_29','fundamental_32','fundamental_33','fundamental_34','fundamental_35',\
'fundamental_36','fundamental_41','fundamental_42','fundamental_43','fundamental_44',\
'fundamental_49','fundamental_61','fundamental_62',\
'technical_0','technical_2','technical_6','technical_9','technical_10','technical_11',\
'technical_12','technical_13','technical_14','technical_16','technical_17','technical_18',\
'technical_22','technical_29','technical_32','technical_33','technical_34','technical_37',\
'technical_38','technical_39','technical_42','technical_43'], axis=1)

nf = 0.1
mean_values = train.median(axis=0)
train.fillna(mean_values, inplace = True)
train = train + nf*np.random.normal(loc=0.0,scale=1.0,size=train.shape)
train = np.clip(train,-2.0,2.0)
#standev = train.std(axis=0)
#train = (train-mean_values)/(standev)


low_y_cut = -0.075
high_y_cut = 0.075
y = observation.train.y
y_is_above_cut = (y > high_y_cut)
y_is_below_cut = (y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)


(xtrain, xval) = train_test_split(train.loc[y_is_within_cut, :])
(ytrain, yval) = train_test_split(y.loc[y_is_within_cut]) 

l = xtrain.shape[1]

#reg = linear_model.Ridge()
reg = ensemble.ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=123, verbose=0)
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
    test = test.drop(['id', 'timestamp','derived_1','derived_3',\
'fundamental_6','fundamental_10','fundamental_12','fundamental_17',\
'fundamental_18','fundamental_19','fundamental_20','fundamental_23','fundamental_26',\
'fundamental_29','fundamental_32','fundamental_33','fundamental_34','fundamental_35',\
'fundamental_36','fundamental_41','fundamental_42','fundamental_43','fundamental_44',\
'fundamental_49','fundamental_61','fundamental_62',\
'technical_0','technical_2','technical_6','technical_9','technical_10','technical_11',\
'technical_12','technical_13','technical_14','technical_16','technical_17','technical_18',\
'technical_22','technical_29','technical_32','technical_33','technical_34','technical_37',\
'technical_38','technical_39','technical_42','technical_43'], axis=1)
    
    #standev = test.std(axis=0)
    #means = test.mean(axis=0)
    test = test.fillna(mean_values, inplace = True)
    #test = np.clip(test,-1.0,1.0)
    #test = (test-means)/(standev)
        
    observation.target.loc[:,'y'] = reg.predict(encoder.predict(test.as_matrix())).clip(low_y_cut, high_y_cut)
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