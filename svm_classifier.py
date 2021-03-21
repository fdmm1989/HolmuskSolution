# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 16:31:28 2021

@author: MengZY
"""

import numpy as np

measure = np.load('X_final.npy')
label = np.load('label.npy')


from sklearn.model_selection import KFold
kf = KFold(n_splits = 5, shuffle = True)

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, classification_report
import time
scaler = preprocessing.StandardScaler().fit(measure)
measure_scaled = scaler.transform(measure)

for train_index, test_index in kf.split(measure_scaled):
    
    X_train, X_test = measure_scaled[train_index], measure_scaled[test_index]
    y_train, y_test = label[train_index], label[test_index]
    start = time.time()
    clf = SVC(kernel = 'rbf', probability = True).fit(X_train, y_train)
    end = time.time()
    print('svm running time: ', end-start)        
    print('svm report: ', classification_report(y_test,clf.predict(X_test)))
    print('svm auc: ', roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))
    
    