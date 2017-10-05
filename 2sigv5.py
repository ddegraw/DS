import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Merge
#from keras.optimizers import SGD, RMSprop, Adam


with pd.HDFStore("C:\\Users\\4126694\\2sig-kaggle\\input\\train.h5", "r") as train:
    # Note that the "train" dataframe is the only dataframe in the file
    df = train.get("train")
    
ids = df["id"].unique()
ids_in = {}
for x in ids:
    time = df[df["id"] == x].timestamp
    if time.min() > 100 and time.max() < 1812:
        ids_in[x] = (time.min(), time.max())

#instrument = 52
#dfi = df[df["id"] == instrument]
pd.set_option('mode.chained_assignment',None)
df.loc[:,"cumprod"] = (1+df["y"]).cumprod()

cols = [x for x in df.columns.values if x not in ["timestamp","y"]]
fund_cols = [c for c in cols if 'fundamental' in c]
tech_cols = [c for c in cols if 'technical' in c]
derv_cols = [c for c in cols if 'derived' in c]

arb = [0,2,6,9,10,11,12,14,17,22,29,32,37,38,39,42,43]
arby_cols = ["fundamental_" + str(s) for s in arb]


y_train = df.pop('y')

l = len(cols)
lfund = len(fund_cols)
ltech = len(tech_cols)
lderv = len(derv_cols)
larb = len(arby_cols)

mean_values = df.mean(axis=0)
df = df.fillna(mean_values, inplace = True)
standev = df.std(axis=0)
df = (df-mean_values)/(standev)


#split the columns
log_data = df[arby_cols]
rest = [item for item in cols if item not in arby_cols]
#scalar data
x_train = df[rest]
#x_train = x_train.drop(['id','timestamp'], axis=1)


logb = Sequential()
logb.add(Dense(128, input_dim=larb))
logb.add(Activation('sigmoid'))
logb.add(Dropout(0.25))
logb.add(Dense(64))
logb.add(Activation('sigmoid'))
logb.add(Dropout(0.25))
logb.add(Dense(16))
logb.add(Activation('relu'))

linb = Sequential()
linb.add(Dense(128, input_dim=l-larb))
linb.add(Activation('relu'))
linb.add(Dropout(0.25))
linb.add(Dense(64))
linb.add(Activation('linear'))
linb.add(Dropout(0.1))
linb.add(Dense(16))
linb.add(Activation('linear'))


merged = Merge([logb, linb], mode='concat')

final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(1, activation='linear'))

final_model.compile(optimizer='Nadam', loss='mse')
final_model.fit([log_data.as_matrix(), x_train.as_matrix()], y_train, batch_size=2000, nb_epoch=5, validation_split=0.2)

#sgd = SGD(lr=0.005, decay=1e-6, momentum=0.1, nesterov=True)
#rms = RMSprop(lr=0.005, rho=0.9, epsilon=1e-08, decay=0.1)
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)

"""
predicted = final_model.predict([log_data.as_matrix(), x_train.as_matrix()])
dataf =  pd.DataFrame(predicted)
dataf.columns = ["predict"]
dataf["input"] = ytrain[:]
dataf.plot(figsize=(15, 5))
"""
#score = model.evaluate(X_test.as_matrix(), y_test, batch_size=16)
#score = final_model.evaluate([log_data.as_matrix(), x_train.as_matrix()], y_train, batch_size=16)