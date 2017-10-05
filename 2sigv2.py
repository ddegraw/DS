import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

#from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import AdaBoostRegressor
#from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

"""
with pd.HDFStore("C:\\Users\\4126694\\2sig-kaggle\\input\\train.h5", "r") as train:
    # Note that the "train" dataframe is the only dataframe in the file
    df = train.get("train")
    
ids = df["id"].unique()
ids_in = {}
for x in ids:
    time = df[df["id"] == x].timestamp
    if time.min() > 100 and time.max() < 1812:
        ids_in[x] = (time.min(), time.max())

instrument = 52
dfi = df[df["id"] == instrument]my

pd.set_option('mode.chained_assignment',None)
dfi.loc[:,"cumprod"] = (1+dfi["y"]).cumprod()

cols = [x for x in dfi.columns.values if x not in ["id", "timestamp","y","cumprod"]]
l = len(cols)

dfj = dfi.fillna(mean_values)
target = dfj.pop('y')
ts = dfj.pop('timestamp')
dfj = dfi.drop(["id","y","cumprod"],axis=1)
dfj=dfj.fillna(0)
features = dfj.values
"""
def _load_data(data, n_prev = 61):  
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY


def train_test_split(data, test_size=0.5):  
    """
    This just splits data to training and testing parts
    """   
    df = pd.DataFrame(data)    
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    tt = df.iloc[0:ntrn]
    vv = df.iloc[ntrn:]
    
    train = np.array(tt)
    val = np.array(vv)


    return (train, val)

(xtrain, xval) = train_test_split(features)
(ytrain, yval) = train_test_split(target) 
(tstrain, tsval) = train_test_split(ts) 


rng = np.random.RandomState(1)
#regr_1 = DecisionTreeRegressor(max_depth=4)
#regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=500, random_state=rng)
#regr_3 = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=5, random_state=rng, loss='ls')
regr_4 = MultiOutputRegressor(RandomForestRegressor(n_estimators=300, max_depth=10, random_state=0))
regr_5 = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=rng)


#regr_1.fit(features, target)
#regr_2.fit(xtrain, ytrain)
#regr_3.fit(xtrain, ytrain)
regr_4.fit(xtrain, ytrain)
regr_5.fit(xtrain, ytrain)

#y_1 = regr_1.predict(features)
#y_2 = regr_2.predict(xval)
#y_3 = regr_3.predict(xval)
y_4 = regr_4.predict(xval)
y_5 = regr_5.predict(xval)


#mse2 = mean_squared_error(yval, y_2)
#mse3 = mean_squared_error(yval, y_3)
mse4 = mean_squared_error(yval, y_4)
mse5 = mean_squared_error(yval, y_5)

print("MSE4: %.6f  MSE5: %.6f" % (mse4,mse5))

plt.figure()
plt.figure(figsize=(15,10))
plt.plot(ts, target,c="k",label="training samples")
plt.plot(tsval, y_4, c="g", label="ADABoost500", linewidth=2)
plt.plot(tsval, y_5, c="r", label="GradBoost500", linewidth=2)

