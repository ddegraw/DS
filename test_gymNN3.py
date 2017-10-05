import kagglegym
import numpy as np
import pandas as pd


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

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
    
    def __init__(self, model, train, log_cols, lin_cols):

        # first save the model ...
        self.model   = model
        self.logc = log_cols
        self.linc = lin_cols
        
        # Get the X, and y values, 
        y = np.array(train.y)
        
        Xlog = train[logc]
        Xlin = train[linc]
        
        self.xMeans = X.mean(axis=0) # Remember to save this value
        self.xStd   = X.std(axis=0)  # Remember to save this value

        Xlog = np.array(Xlog.fillna( self.xMeans ))
        Xlog = (Xlog - np.array(self.xMeans))/np.array(self.xStd)
        
        Xlin = np.array(Xlin.fillna( self.xMeans ))
        Xlin = (Xlin - np.array(self.xMeans))/np.array(self.xStd)
        
        # fit the model
        self.model.fit(Xlog, Xlin, y)
        
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
        
def checkModel(modelToUse, log_cols, lin_cols):
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
    logtrain   = train[['timestamp', 'y'] + log_cols]
    logtrain   = logtrain.groupby('timestamp').aggregate(np.mean)
    logtrain.y = np.cumsum(logtrain.y) # easier to visualize
    
    lintrain   = train[['timestamp', 'y'] + lin_cols]
    lintrain   = lintrain.groupby('timestamp').aggregate(np.mean)
    lintrain.y = np.cumsum(lintrain.y) # easier to visualize
    
    
    print('fitting a model')
    model = fitModel(modelToUse, train, log_cols, lin_cols)
    
    print('predict the same data')
    yHat = model.predict(train) # We already select required columns
    
    plt.figure()
    plt.plot(yHat, color='black', lw=2, label='predicted')
    plt.plot(train.y, '.', mec='None', mfc='orange', label='original')
    plt.legend(loc='lower right')
    
    return        

def getScore(modelToUse, log_cols, lin_cols):
    
    print('Starting a new calculation for score')
    rewards = []
    env = kagglegym.make()
    observation = env.reset()
    
    print('fitting a model')
    model = fitModel(modelToUse, observation.train.copy(), log_cols, lin_cols)

    print('Starting to fit a model')
    while True:
        
        prediction  = model.predict(observation.features.copy())
        target      = observation.target
        target['y'] = prediction
        
        timestamp = observation.features["timestamp"][0]
        if timestamp % 100 == 0:
            print(timestamp)

        observation, reward, done, info = env.step(target)
        rewards.append(reward)
        if done: break
            
    return info['public_score'], rewards


checkModel(LinearRegression(), [c for c in allCols if 'fundamental' in c])
plt.title('fundamentals')

checkModel(LinearRegression(), [c for c in allCols if 'technical' in c])
plt.title('technicals')
fitting a model
