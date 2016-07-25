# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 16:33:51 2016

@author: weinfz
"""

import pandas as pd 
    
import numpy as np
import sklearn.ensemble
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.externals import joblib

headers = []
for x in range(8):
    for y in range(8):
        headers.append(str(x)+str(y))
headers.append('first')
headers.append('second')
data = pd.read_csv('data1.csv',header=None,names=headers)
first_win = (data['first'] > data['second']).astype(int)
first_win.loc[data['first'] < data['second']] = -1

data.drop(['first','second'],axis=1,inplace=True)

X_train, X_test, y_train, y_test = train_test_split(data, first_win, test_size=0.33, random_state=42)

rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)

weights = [1+(i/len(y_train)) for i in range(len(y_train))]
rf.fit(X_train, y_train)
pred = rf.predict_proba(X_test)
if pred.shape[1] == 3:
    pred = np.delete(pred,1,1)

#scores = cross_val_score(rf, X_test, y_test)
#rf.predict(test_X)
y_test[y_test==0] = -1
print(roc_auc_score(y_test,pred[:,1]))
joblib.dump(rf, 'rf1.pkl',compress=True)
print('done') 

importances = rf.feature_importances_
importances = importances.reshape([8,8])





