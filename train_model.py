from keras.callbacks import EarlyStopping, ModelCheckpoint

VALIDATION_SPLIT = 0.1
re_weight = True

if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None
    

#Retraining
#bst_model_path = STAMP + '.h5'
#model.load_weights(bst_model_path)
#STAMP = STAMP+'_retrain'
bst_model_path = STAMP + '.h5'
early_stopping =EarlyStopping(monitor='val_loss', patience=2)
#bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([data_1_train,data_2_train], labels_train, validation_data=([data_1_val, data_2_val], labels_val, weight_val), nb_epoch=5, batch_size=256, shuffle=True,class_weight=class_weight,callbacks=[early_stopping, model_checkpoint])
  
model.load_weights(bst_model_path)
bst_val_score=min(hist.history['val_loss'])

preds = model.predict([test_data_1, test_data_2],batch_size=2048, verbose=1)
preds += model.predict([test_data_2, test_data_1], batch_size=2048, verbose=1)
preds /= 2.0
print(bst_val_score)

out_df = pd.DataFrame({"test_id":test_labels, "is_duplicate":preds.ravel()})
out_df.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)
del out_df