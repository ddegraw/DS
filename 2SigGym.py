import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import kagglegym

class fitModel():
    '''
        This class is going to take train values
        and a particular type of model and take care of
        the prediction step and wil contain a fit
        step. 
        
        Remember to send in a copy of train because
        there is a high chance that it will be modified
        
        the model is a sklearn model like ElasticNetCV
        
        all other parameters are passed onto the model
    '''
    
    def __init__(self, model, train, columns):

        # first save the model ...
        self.model   = model
        self.columns = columns
        
        # Get the X, and y values, 
        y = np.array(train.y)
        
        X = train[columns]
        self.xMeans = X.mean(axis=0) # Remember to save this value
        self.xStd   = X.std(axis=0)  # Remember to save this value

        X = np.array(X.fillna( self.xMeans ))
        X = (X - np.array(self.xMeans))/np.array(self.xStd)
        
        # fit the model
        self.model.fit(X, y)
        
        return
    
    def predict(self, features):
        '''
            This function is going to return the predicted
            value of the function that we are trying to 
            predict, given the observations. 
        '''
        X = features[self.columns]
        X = np.array(X.fillna( self.xMeans ))
        X = (X - np.array(self.xMeans))/np.array(self.xStd)

        return self.model.predict(X)
def checkModel(modelToUse, columns):
    '''
        This  function checks and makes sure that the 
        model provided is doing what it is supposed to
        do. This is a sanity check ...
    '''
    
    rewards = []
    env = kagglegym.make()
    observation = env.reset()
    
    train = observation.train
    
    # Just to make things easier to visualize
    # and also to speed things up ...
    # -----------------------------------------
    train   = train[['timestamp', 'y'] + columns]
    train   = train.groupby('timestamp').aggregate(np.mean)
    train.y = np.cumsum(train.y) # easier to visualize
    
    print('fitting a model')
    model = fitModel(modelToUse, train, columns)
    
    print('predict the same data')
    yHat = model.predict(train) # We already select required columns
    
    plt.figure()
    plt.plot(yHat, color='black', lw=2, label='predicted')
    plt.plot(train.y, '.', mec='None', mfc='orange', label='original')
    plt.legend(loc='lower right')
    
    return
    
def getScore(modelToUse, columns):
    
    print('Starting a new calculation for score')
    rewards = []
    env = kagglegym.make()
    observation = env.reset()
    
    print('Fitting the model')
    model = fitModel(modelToUse, observation.train.copy(), columns)

    print('Running the test set')
    while True:
        
        prediction  = model.predict(observation.features.copy())
        target      = observation.target
        target['y'] = prediction
        
        timestamp = observation.features["timestamp"][0]
        if timestamp % 100 == 0:
            print(timestamp)
            print(reward)

        observation, reward, done, info = env.step(target)
        rewards.append(reward)
        if done: break
            
    return info['public_score'], rewards

env     = kagglegym.make()
"""
allCols = env.reset().train.columns
allCols = allCols.drop(['id', 'y', 'timestamp'])
cat = [0,2,6,9,10,11,12,14,16,17,18,22,29,32,34,37,38,39,42,43]
cat_cols = ["technical_" + str(s) for s in cat]
rest = [item for item in allCols if item not in cat_cols]
cols = [c for c in allCols if 'technical' in c] #derived, technical
"""


score = getScore(ElasticNetCV(), rest)[0]
print score