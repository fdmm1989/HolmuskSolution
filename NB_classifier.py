# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 21:55:17 2021

@author: MengZY
"""

import numpy as np

measure = np.load('X_final.npy')
label = np.load('label.npy')


from sklearn.model_selection import KFold
kf = KFold(n_splits = 5, shuffle = True)


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, classification_report
import time

for train_index, test_index in kf.split(measure):
    
    X_train, X_test = measure[train_index], measure[test_index]
    y_train, y_test = label[train_index], label[test_index]
    start = time.time()
    clf = MultinomialNB().fit(X_train, y_train)
    end = time.time()
    print('GauNB time: ', end-start)        
    print('GauNB report: ', classification_report(y_test,clf.predict(X_test)))
    print('GauNB auc: ', roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))
