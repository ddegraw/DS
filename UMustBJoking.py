import kagglegym
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cluster import KMeans

env = kagglegym.make()
o = env.reset()

#o.train = o.train[:1000]
train = o.train
ymean_dict = dict(o.train.groupby(["id"])["y"].median())
d_mean= o.train.median(axis=0)
train = train.fillna(d_mean)
cols = ['technical_30', 'technical_20', 'fundamental_11']
train2 = train[cols].copy()

y2 = train.y.copy()

N_timestamps = 450 # how many timestamps that are non-nan for each id 450 initially
N_corr_cut = 0.0 # min mean correlation coefficient when dropping id's
select_ids = train[["id","y"]].groupby("id").count()
selected_ids = select_ids[select_ids.y > N_timestamps]#== N_timestamps]
selected_ids = np.array(selected_ids.index)
index_ids = [i in selected_ids for i in train.id]
data_corr = train[index_ids][["id","timestamp","y"]].copy()
index_ids = [i in selected_ids for i in o.train.id]
data_corr_full = o.train[index_ids][["id","timestamp","y"]].copy()
df_id = data_corr[['id', 'timestamp', 'y']].pivot_table(values='y',
                                     index='timestamp',columns='id')
df_id_full = data_corr_full[['id', 'timestamp', 'y']].pivot_table(values='y',
                                     index='timestamp',columns='id')
df_id_cumsum = df_id.cumsum()
diff = df_id_cumsum.mean(axis=1)
df_id_cumsum = df_id_cumsum.subtract(diff.values,axis="rows")
## Full Data
df_id_cumsum_full = df_id_full.cumsum()
diff = df_id_cumsum_full.mean(axis=1)
df_id_cumsum_full = df_id_cumsum_full.subtract(diff.values,axis="rows")
corr_cumsum = df_id_cumsum.corr()
dist = corr_cumsum.as_matrix()
dist_id_mean = np.mean(np.abs(dist),axis = 1)
index_mean = dist_id_mean > N_corr_cut

tmp_cut = dist[index_mean,:]
tmp_cut = tmp_cut[:,index_mean]
## Perform Kmeans to easily get the two clusters
clf = KMeans(n_clusters = 2)
clf.fit(tmp_cut)
labels = clf.labels_

train['cumsum'] = train['id'].map(ymean_dict)
for ind, row in train.iterrows():
    if row['id'] in df_id_cumsum_full.columns:
        train.set_value(ind, 'cumsum', df_id_cumsum_full[row['id']][row['timestamp']])

sidedict = dict(zip(selected_ids,labels))
train['side'] = train['id'].map(sidedict)


del data_corr
del data_corr_full
del df_id
del df_id_full
del df_id_cumsum
del df_id_cumsum_full
del diff
del corr_cumsum
del dist_id_mean
del dist
del tmp_cut

y = train[train['id'].isin(selected_ids)].y.copy()

low_y_cut = -0.075
high_y_cut = 0.075
y_is_above_cut = (y2 > high_y_cut)
y_is_below_cut = (y2 < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

train['side'] = train['side'].fillna(-1)
#train = np.clip(train,-2.2,2.2)
n = train.isnull().sum(axis=1)
for c in train.columns:
    train[c + '_nan_'] = pd.isnull(train[c])
    d_mean[c + '_nan_'] = 0
train = train.fillna(d_mean)
train['znull'] = n
n = []

train.pop('y')
train.pop('y_nan_')

rfr = ExtraTreesRegressor(n_estimators=1, max_depth=4, n_jobs=-1, random_state=123, verbose=0)
#model1 = rfr.fit(np.array(train.loc[y_is_within_cut,:].values), y.loc[y_is_within_cut])
model1 = rfr.fit(np.array(train[train['id'].isin(selected_ids)]), y)

#https://www.kaggle.com/bguberfain/two-sigma-financial-modeling/univariate-model-with-clip/run/482189

cols = ['technical_30', 'technical_20', 'fundamental_11']
#cols = ['technical_30']
train2 = train[cols]
model2 = Ridge()
model2.fit(np.array(train2.loc[y_is_within_cut,:].values), y2.loc[y_is_within_cut])

#https://www.kaggle.com/ymcdull/two-sigma-financial-modeling/ridge-lb-0-0100659


while True:
    test = o.features
    test = test.fillna(d_mean)
    test['side'] = test['id'].map(sidedict)
    test['side'] = test['side'].fillna(-1)
    n = test.isnull().sum(axis=1)
    for c in test.columns:
        test[c + '_nan_'] = pd.isnull(test[c])
    test = test.fillna(d_mean)
    test['znull'] = n
    excl = ['y', 'y_nan_']
    col = [c for c in train.columns if c not in excl]    
    
    
    test = test[col]
    pred = o.target
    test2 = test[cols].as_matrix()
    #test2 = np.array(test2.fillna(d_mean).values).reshape(-1,1)
    lamb = 0.4
    pred['y'] = (model1.predict(test).clip(low_y_cut, high_y_cut) * lamb) + (model2.predict(test2).clip(low_y_cut, high_y_cut) * (1.0-lamb))
    pred['y'] = pred.apply(lambda r: 0.95 * r['y'] + 0.05 * ymean_dict[r['id']] if r['id'] in ymean_dict else r['y'], axis = 1)
    pred['y'] = [float(format(x, '.6f')) for x in pred['y']]
    o, reward, done, info = env.step(pred)
    if done:
        print("el fin ...", info["public_score"])
        break
    if o.features.timestamp[0] % 100 == 0:
        print(reward)
        
print(info)
