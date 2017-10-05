import kagglegym
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.tsa.stattools as ts 

env = kagglegym.make()
o = env.reset()

# o.train = o.train[:1000]
# excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
# col = [c for c in o.train.columns if c not in excl]

train = o.train
ymean_dict = dict(o.train.groupby(["id"])["y"].median())
d_mean= o.train.median(axis=0)
train = train.fillna(d_mean)

N_timestamps = 100 # how many timestamps that are non-nan for each id
#N_corr_cut = 0.4 # min mean correlation coefficient when dropping id's
select_ids = train[["id","y"]].groupby("id").count()
selected_ids = select_ids[select_ids.y > N_timestamps]#== N_timestamps]
selected_ids = np.array(selected_ids.index)
#added below line for version 5
#train = train[train['id'].isin(selected_ids)]
select_ids = train[["id","y"]].groupby("id").count()
selected_ids = np.array(select_ids.index)
index_ids = [i in selected_ids for i in train.id]
data_corr_full = o.train[index_ids][["id","timestamp","y"]].copy()
df_id_full = data_corr_full[['id', 'timestamp', 'y']].pivot_table(values='y',index='timestamp',columns='id')
df_id_cumsum_full = df_id_full.cumsum()

dick = pd.DataFrame(0, index=np.arange(len(df_id_cumsum_full.columns)),columns=df_id_cumsum_full.columns)
stat = pd.DataFrame(0, index=np.arange(len(df_id_cumsum_full.columns)),columns=df_id_cumsum_full.columns)
res = []

for col1 in df_id_full.columns:
    for col2 in df_id_full.columns:
        if col1 != col2:
            res = LinearRegression()
            y1_null = np.isnan(df_id_cumsum_full[col1])
            y2_null = np.isnan(df_id_cumsum_full[col2])        
            both_notnull = (~y1_null & ~y2_null)
            if sum(both_notnull) > 100:
                res.fit(df_id_cumsum_full[col1].loc[both_notnull].reshape(-1, 1), df_id_cumsum_full[col2].loc[both_notnull])        
                beta_hr = res.coef_[0]
                res = df_id_cumsum_full[col2].loc[both_notnull] - beta_hr*df_id_cumsum_full[col1].loc[both_notnull]
                dick[col1][col2] = ts.adfuller(res)[0] - ts.adfuller(res)[4]['5%']
                #stat[col1][col2] = ts.adfuller(res)[4]['5%']
                #dick[col1][col2] = ts.coint(np.array(df_id_full[col1]),np.array(df_id_full[col2]))
       