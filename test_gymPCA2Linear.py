import kagglegym
import numpy as np
import pandas as pd

from sklearn import decomposition, linear_model, ensemble
from sklearn import manifold


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

"""
#Transform data matrix
pca = decomposition.PCA(n_components=40)
pca.fit(train.as_matrix())
train = pca.fit_transform(train.as_matrix())
l = train.shape[1]
"""

tsne = manifold.TSNE(n_components=5, init='pca', random_state=0)
train = tsne.fit_transform(train.as_matrix())
l = train.shape[1]

reg = ensemble.ExtraTreesRegressor(n_estimators=100,max_depth=4,n_jobs = 14)
#reg = linear_model.LinearRegression(fit_intercept=False, normalize=True, copy_X=True,n_jobs = -1)
reg.fit(train,y.loc[y_is_within_cut])

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
    target.loc[:,'y'] = reg.predict(pca.fit_transform(test.as_matrix()))
    #observation.target.fillna(0, inplace=True)
    
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        print(reward)

    observation, reward, done, info = env.step(target)
    if done:
        break
    
print(info)