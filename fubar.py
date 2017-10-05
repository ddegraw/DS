
m.load_weights(bst_model_path)
bst_val_score=min(hist.history['val_loss'])
preds = m.predict([test_data_1, test_data_2, aux_test],batch_size=2048, verbose=1)
preds += m.predict([test_data_2, test_data_1, aux_test], batch_size=2048, verbose=1)
preds /= 2.0
print(preds.shape)

out_df = pd.DataFrame({"test_id":test_labels, "is_duplicate":preds.ravel()})
out_df.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)
del out_df
