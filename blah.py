x['senti1'] = df['question1'].apply( lambda x: sid.polarity_scores(x).get('compound'))
x['senti2'] = df['question2'].apply( lambda x: sid.polarity_scores(x).get('compound'))
x['polar'] = (np.sign(x['senti1']) == np.sign(x['senti2'])).astype(int) 

print(x.columns)
    #print(x.describe())

np.random.seed(1234)
    
sz_train = df_train.shape[0]
perm = np.random.permutation(sz_train)
x_test  = x[sz_train:]
x_train = x[:sz_train]

idx_train = perm[:int(sz_train*(1-VALIDATION_SPLIT))]
x_t = pd.concat([x_train.ix[idx_train], x_train.ix[idx_train]])
    
idx_val = perm[int(sz_train*(1-VALIDATION_SPLIT)):]
x_val = pd.concat([x_train.ix[idx_val], x_train.ix[idx_val]])
    
    
x_t = x_t.fillna(0)
x_test = x_test.fillna(0)
x_val = x_val.fillna(0)
    
ss = StandardScaler()
ss.fit(np.vstack((x_t, x_test,x_val)))
x_t = ss.transform(x_t)
x_test = ss.transform(x_test)
x_val = ss.transform(x_val)

aux_test[['polar']] = x_test[['polar']]
aux_train[['polar']] = x_t[['polar']]
aux_val[['polar']] = x_val[['polar']]
