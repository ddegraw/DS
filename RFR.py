# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 08:46:46 2017

@author: 4126694
"""

#from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

STAMP = "GBmodel"
class_weight = {0: 1.309028344, 1: 0.472001959}

gbr = GradientBoostingClassifier(n_estimators=300, max_depth=4, random_state=123, verbose=1)
#rfr = ExtraTreesClassifier(n_estimators=300, max_depth=4, n_jobs=14, random_state=123, verbose=1,class_weight=class_weight)
#model1 = rfr.fit(np.array(train.loc[y_is_within_cut,:].values), y.loc[y_is_within_cut])
model1 = gbr.fit(aux_train, labels_train)
sco = gbr.score(aux_val,labels_val)

print(sco)
predi = model1.predict(aux_test)

out_df3 = pd.DataFrame({"test_id":test_ids, "is_duplicate":predi.ravel()})
out_df3.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)


