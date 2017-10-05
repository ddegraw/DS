import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn import linear_model, decomposition
from sklearn.feature_selection import SelectKBest, f_regression, SelectPercentile, mutual_info_regression, SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import manifold, svm, ensemble

import kagglegym

# The "environment" is our interface for code competitions

env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Get the train dataframe
train = observation.train
#mean_values = train.mean(axis=0)
#standev = train.std(axis=0)

y = train.pop('y')

mean_values = train.mean(axis=0)
standev = train.std(axis=0)
train.fillna(mean_values, inplace=True)
train = (train-mean_values)/(standev)

train = train.drop(['id', 'timestamp','fundamental_61','derived_1','fundamental_17','derived_4','fundamental_26','fundamental_23','fundamental_1'], axis=1)
#low_y_cut = -0.086093
#high_y_cut = 0.093497

low_y_cut = -0.086093
high_y_cut = 0.093497

y_is_above_cut = (y > high_y_cut)
y_is_below_cut = (y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

#regression = linear_model.ElasticNetCV()
#pca = decomposition.PCA(n_components=10)
#tsne = manifold.LocallyLinearEmbedding(n_neighbors = 200, n_components = 16, method='hessian')
#pipe = Pipeline(steps=[('pca', pca), ('regression', regression)])

train = train.loc[y_is_within_cut,:]
reg = ensemble.ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=123, verbose=0)
anova = SelectKBest(f_regression,k=6)
#anova = SelectFromModel(reg,threshold=.9)
#X = tsne.fit_transform(train.as_matrix())

#reg = svm.svr(kernel='linear')
pipl = make_pipeline(anova,reg)
pipl.fit(train,y.loc[y_is_within_cut])

#pca.fit(X.as_matrix())
#reg.fit(pca.fit_transform(X.as_matrix()),Y.as_matrix())
#tsne.fit(train.as_matrix())
#reg.fit(X,y.loc[y_is_within_cut])

del train

print('Running the test set')
while True:
    #print('Running for test.')
    target = observation.target
    test = observation.features
 
    means = observation.features.mean(axis=0)
    standev = test.std(axis=0)
    test = observation.features.fillna(means)
    test = (test-mean_values)/(standev)       
    
    test = test.drop(['id','timestamp','fundamental_61','derived_1','fundamental_17','derived_4','fundamental_26','fundamental_23','fundamental_1'], axis=1)       
    #test_x = np.array(observation.features[col].values).reshape(-1,1)
    #target.loc[:,'y'] = reg.predict(pca.fit_transform(xtest.as_matrix())).clip(low_y_cut, high_y_cut)
    
    target.loc[:,'y'] = pipl.predict(test).clip(low_y_cut, high_y_cut)
    #observation.target.fillna(0, inplace=True)
    
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        print reward

    observation, reward, done, info = env.step(target)
    if done:
        break
    
print(info)
