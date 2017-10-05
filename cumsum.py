import kagglegym
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

env = kagglegym.make()
o = env.reset()

# o.train = o.train[:1000]
# excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
# col = [c for c in o.train.columns if c not in excl]

train = o.train
ymean_dict = dict(o.train.groupby(["id"])["y"].median())
d_mean= o.train.median(axis=0)
train = train.fillna(d_mean)

N_timestamps = 450 # how many timestamps that are non-nan for each id
#N_corr_cut = 0.4 # min mean correlation coefficient when dropping id's
select_ids = train[["id","y"]].groupby("id").count()
selected_ids = select_ids[select_ids.y > N_timestamps]#== N_timestamps]
selected_ids = np.array(selected_ids.index)
#added below line for version 5
train = train[train['id'].isin(selected_ids)]
#select_ids = train[["id","y"]].groupby("id").count()
#selected_ids = np.array(select_ids.index)
#index_ids = [i in selected_ids for i in train.id]
#data_corr_full = o.train[index_ids][["id","timestamp","y"]].copy()
#df_id_full = data_corr_full[['id', 'timestamp', 'y']].pivot_table(values='y',index='timestamp',columns='id')
#df_id_cumsum_full = df_id_full.cumsum()

#train['cumsum'] = train['id'].map(ymean_dict)
#for ind, row in train.iterrows():
#    if row['id'] in df_id_cumsum_full.columns:
#        train.set_value(ind, 'cumsum', df_id_cumsum_full[row['id']][row['timestamp']])

y = train.pop('y')
#y = train.pop('cumsum')
train = train.drop(['timestamp','id'],axis=1)

low_y_cut = -0.075
high_y_cut = 0.075
y_is_above_cut = (y > high_y_cut)
y_is_below_cut = (y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

low_y1_cut = -0.086093
high_y1_cut = 0.093497
y1_is_above_cut = (y > high_y_cut)
y1_is_below_cut = (y < low_y_cut)
y1_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

n = train.isnull().sum(axis=1)
for c in train.columns:
    train[c + '_nan_'] = pd.isnull(train[c])
    d_mean[c + '_nan_'] = 0
train = train.fillna(d_mean)
train['znull'] = n
#train = np.clip(train,-2.2,2.2)

m1cols = ['technical_14', 'technical_34','technical_10','technical_22','technical_35','technical_36','technical_27','technical_30',\
'technical_7','technical_19','technical_40','technical_21','technical_29',\
'fundamental_54','fundamental_59','fundamental_60','fundamental_18','fundamental_56','fundamental_7','fundamental_51','fundamental_21',\
'fundamental_53','fundamental_10','fundamental_38','fundamental_15','fundamental_5','fundamental_45','fundamental_55','fundamental_43',\
'fundamental_34','fundamental_2','fundamental_20']

#rfr = ExtraTreesRegressor(n_estimators=400, max_depth=7, max_features=30, n_jobs=-1, random_state=17, verbose=0)
#model1 = rfr.fit(np.array(train.loc[y_is_within_cut,:].values), y.loc[y_is_within_cut])
#model1 = rfr.fit(np.array(train.loc[y1_is_within_cut,:].values), y.loc[y1_is_within_cut])
model1 = rfr.fit(np.array(train[m1cols].values), y)

m2cols = ['technical_20','technical_30','fundamental_11']
#m2cols = ['technical_20']
train2 = train[m2cols]
model2 = Ridge()
model2.fit(np.array(train2.loc[y_is_within_cut,:].values), y.loc[y_is_within_cut])



"""
while True:
    test = o.features
    test = test.fillna(d_mean)
    test = test.drop(['timestamp','id'],axis=1)
    n = test.isnull().sum(axis=1)
    for c in test.columns:
        test[c + '_nan_'] = pd.isnull(test[c])
    test = test.fillna(d_mean)
    test['znull'] = n    
    pred = o.target
    #test2 = np.array(test2.fillna(d_mean).values).reshape(-1,1)
    lamb = 0.65
    pred['y'] = (model1.predict(test[m1cols]).clip(low_y_cut, high_y_cut) * lamb) + (model2.predict(test[m2cols]).clip(low_y_cut, high_y_cut) * (1.0-lamb))
    pred['y'] = pred.apply(lambda r: 0.95 * r['y'] + 0.05 * ymean_dict[r['id']] if r['id'] in ymean_dict else r['y'], axis = 1)
    pred['y'] = [float(format(x, '.6f')) for x in pred['y']]
    o, reward, done, info = env.step(pred)
    if done:
        print("el fin ...", info["public_score"])
        break
    if o.features.timestamp[0] % 100 == 0:
        print(reward)
        
print(info)

"""