# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:12:24 2017

@author: 4126694
"""
from keras.callbacks import EarlyStopping, ModelCheckpoint

STAMP = 'lstm_96_48_0.18_0.17'
bst_model_path = STAMP + '.h5'

model.load_weights(bst_model_path)

early_stopping =EarlyStopping(monitor='val_loss', patience=2)

STAMP = 'lstm_96_48_0.18_0.17_retrain'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([data_2,data_1], labels, validation_split=VALIDATION_SPLIT, nb_epoch=10, batch_size=1024, shuffle=True,class_weight=class_weight,callbacks=[early_stopping, model_checkpoint])
  
bst_val_score=min(hist.history['val_loss'])
preds = model.predict([test_data_1, test_data_2],batch_size=4096, verbose=1)
preds += model.predict([test_data_2, test_data_1], batch_size=4096, verbose=1)
preds /= 2.0
print(preds.shape)

out_df = pd.DataFrame({"test_id":test_labels, "is_duplicate":preds.ravel()})
out_df.to_csv('%.4f_'%(bst_val_score)+STAMP+'_retrain.csv', index=False)
del out_df