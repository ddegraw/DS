import kagglegym
import numpy as np
import pandas as pd

from sklearn import decomposition, linear_model, ensemble
from sklearn import manifold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

# The "environment" is our interface for code competitions
env = kagglegym.make()
o = env.reset()

#col = ['technical_22','technical_20', 'technical_30', 'technical_13', 'technical_34']
col = ['technical_20']
train = o.train
d_mean= train.median(axis=0)
train = train.sample(3000)
y = train['y'].pop
train = train.drop(['id', 'timestamp','fundamental_61','derived_1','fundamental_17','derived_4','fundamental_26','fundamental_23','fundamental_1'], axis=1)

d_mean= train.median(axis=0)

#nullcount = train.count().sum().sort(axis=0, ascending=False, inplace=False)

#train = train[col]
n = train.isnull().sum(axis=1)
for c in train.columns:
    train[c + '_nan_'] = pd.isnull(train[c])
    d_mean[c + '_nan_'] = 0
train = train.fillna(d_mean)
train['znull'] = n
n = []

low_y_cut = -0.086093
high_y_cut = 0.093497
y_is_above_cut = (y > high_y_cut)
y_is_below_cut = (y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

full_df_y_std = 0.022421712
full_df_y_mean = 0.00022174742
full_df_y_var = 0.00050273316

coll = ['technical_22','technical_20', 'technical_30_nan_', 'technical_20_nan_', 'technical_30', 'technical_13', 'technical_34']
#kernel = C(0.02, (1e-3, 1e-1)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-5, 1e-1))
gp_kernel_1 = 3.0e0*full_df_y_std * RBF(length_scale=1e-4, length_scale_bounds=(1e-4, 1e3))
gp_kernel_2 = 1.0e3 * RBF(length_scale=1.0, length_scale_bounds=(3e0, 3e3))
gp_kernel_3 = WhiteKernel(noise_level=4e0*full_df_y_var)
    
kernel = gp_kernel_1 + gp_kernel_2 + gp_kernel_3

model2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=6)
model2.fit(np.array(train[coll].fillna(d_mean).loc[y_is_within_cut,:].values), y.loc[y_is_within_cut])

"""
#Transform data matrix
pca = decomposition.PCA(n_components=40)
pca.fit(train.as_matrix())
train = pca.fit_transform(train.as_matrix())
l = train.shape[1]
"""

#tsne = manifold.TSNE(n_components=5, init='pca', random_state=0)
#train = tsne.fit_transform(train.as_matrix())
#l = train.shape[1]

#reg = ensemble.ExtraTreesRegressor(n_estimators=100,max_depth=4,n_jobs = 14)
#reg = linear_model.LinearRegression(fit_intercept=False, normalize=True, copy_X=True,n_jobs = -1)
#reg.fit(train,y.loc[y_is_within_cut])

del train

print('Running for test.')
while True:

    test = o.features[coll]
    n = test.isnull().sum(axis=1)
    for c in test.columns:
        test[c + '_nan_'] = pd.isnull(test[c])
    test = test.fillna(d_mean)
    test['znull'] = n
    #test2 = np.array(o.features[coll].fillna(d_mean).values)
    #test_x = np.array(observation.features[col].values).reshape(-1,1)
    target.loc[:,'y'] = model2.predict(test).clip(low_y_cut, high_y_cut)
    #observation.target.fillna(0, inplace=True)
    
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        print(reward)

    observation, reward, done, info = env.step(target)
    if done:
        break
    
print(info)