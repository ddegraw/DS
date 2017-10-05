# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 09:43:58 2017

@author: 4126694
"""
out_df = pd.read_csv('./0.2802_lstm_96_48_0.18_0.17.csv')
out_df.quantile(.05)
bottom5 = out_df[out_df['is_duplicate'] <= 0.00003]
testo5b = testo[testo.test_id.isin(bottom5.test_id)]
testo5b.to_csv(BASE_DIR+'bottom5.csv', index=False)
