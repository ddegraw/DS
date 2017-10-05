import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#from sklearn.linear_model import ElasticNetCV, LinearRegression
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.svm import LinearSVR
from sklearn import linear_model, decomposition
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import kagglegym

# The "environment" is our interface for code competitions

env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Get the train dataframe
train = observation.train
#mean_values = train.mean(axis=0)
#standev = train.std(axis=0)
"""
y = train.pop('y')

mean_values = train.mean(axis=0)
standev = train.std(axis=0)
train.fillna(mean_values, inplace=True)
train = (train-mean_values)/(standev)

xtrain = train.drop(['id', 'timestamp'], axis=1)

low_y_cut = -0.086093
high_y_cut = 0.093497

y_is_above_cut = (y > high_y_cut)
y_is_below_cut = (y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

del train
"""
#regression = linear_model.ElasticNetCV()
pca = decomposition.PCA(n_components=10)
pipe = Pipeline(steps=[('pca', pca), ('regression', regression)])
"""
pca.fit(xtrain)

n_components = [2, 4, 8, 16, 20, 40]
Cs = np.arange(0.1, 1, 0.1)

#Parameters of pipelines can be set using ‘__’ separated parameter names:

estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              regression__l1_ratio=Cs))
estimator.fit(xtrain, y)

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()
"""
#reg = GradientBoostingRegressor(n_estimators=100, max_depth=3, min_samples_split=2,learning_rate=0.01, loss='ls',max_features=80)

#reg = LinearSVR(C=1.0, max_iter = 200, verbose=True)

#pipe.set_params(pca__n_components=20).fit(xtrain.as_matrix(),y)

X = xtrain.loc[y_is_within_cut,:]
Y = y.loc[y_is_within_cut]

reg = linear_model.LinearRegression()
reg.fit(pca.fit_transform(X),Y)


print('Running the test set')
while True:
    #print('Running for test.')
    target = observation.target
    test = observation.features
 
    means = observation.features.mean(axis=0)
    standev = test.std(axis=0)
    test = observation.features.fillna(means)
    test = (test-mean_values)/(standev)       
    
    xtest = test.drop(['id', 'timestamp'], axis=1)
           
    #test_x = np.array(observation.features[col].values).reshape(-1,1)
    target.loc[:,'y'] = reg.predict(pca.fit_transform(xtest)).clip(low_y_cut, high_y_cut)
    #observation.target.fillna(0, inplace=True)
    
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        print reward

    observation, reward, done, info = env.step(target)
    if done:
        break
    
print(info)
