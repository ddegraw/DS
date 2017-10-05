import kagglegym
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression,Ridge
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

env = kagglegym.make()
o = env.reset()
train = o.train
#excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
excl = ['id','timestamp','y','derived_1','fundamental_1','fundamental_17','fundamental_26','fundamental_33','fundamental_1','fundamental_42','fundamental_61']
col = [c for c in train.columns if c not in excl]
oobcol = ['technical_20','technical_19','technical_43','technical_36','technical_30','fundamental_27','fundamental_11','fundamental_55','fundamental_12','fundamental_56']
regcol =['technical_20','fundamental_11']


train = train[col]
d_mean= train[col].median(axis=0)

n = train.isnull().sum(axis=1)
for c in train.columns:
    train[c + '_nan_'] = pd.isnull(train[c])
    d_mean[c + '_nan_'] = 0
#train['znull'] = n
n = []
   
train.fillna(d_mean, inplace = True)
stad = train.std(axis=0)
train = (train-d_mean)/stad

#rfr = ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=123, verbose=0)
#model1 = rfr.fit(train, o.train['y'])
l = train.shape[1]
#l = len(oobcol)
model = Sequential()
model.add(Dense(256, input_dim=l))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(16))
model.add(Activation('linear'))
model.add(Dense(1))  
Activation('linear')

model.compile(optimizer='adam', loss='mse')
model.fit(train.as_matrix(), o.train['y'], batch_size=2000, nb_epoch=3, verbose = 1, validation_split=0.2) 

#https://www.kaggle.com/bguberfain/two-sigma-financial-modeling/univariate-model-with-clip/run/482189
low_y_cut = -0.075
high_y_cut = 0.075
y_is_above_cut = (o.train.y > high_y_cut)
y_is_below_cut = (o.train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
model2 = Ridge()
model2.fit(np.array(train[col].loc[y_is_within_cut,:].values), o.train.loc[y_is_within_cut, 'y'])
#train = []

#https://www.kaggle.com/ymcdull/two-sigma-financial-modeling/ridge-lb-0-0100659
ymean_dict = dict(o.train.groupby(["id"])["y"].median())

while True:
    test = o.features
    test = test[col]
    #stad = test[col].std(axis=0)
    pred = o.target
    n = test.isnull().sum(axis=1)
    for c in test.columns:
        test[c + '_nan_'] = pd.isnull(test[c])
        d_mean[c + '_nan_'] = 0
    #test['znull'] = n    
    test = test.fillna(d_mean,inplace = True)
    stad = test.std(axis=0)
    test = (test-d_mean)/stad
    lamb = 0.5
    test2 = np.array(o.features[regcol].values).reshape(-1,1)
    pred['y'] = (model.predict(test.values).clip(low_y_cut, high_y_cut) * (1-lamb)) + (model2.predict(test2).clip(low_y_cut, high_y_cut) * lamb)
    pred['y'] = pred.apply(lambda r: 0.95 * r['y'] + 0.05 * ymean_dict[r['id']] if r['id'] in ymean_dict else r['y'], axis = 1)
    pred['y'] = [float(format(x, '.6f')) for x in pred['y']]
    o, reward, done, info = env.step(pred)
    if done:
        print(info["public_score"])
        break
    if o.features.timestamp[0] % 100 == 0:
        print(reward)