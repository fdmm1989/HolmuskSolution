# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 16:32:38 2021

@author: MengZY
"""
import numpy as np

measure = np.load('X_final.npy')
label = np.load('label.npy')


from sklearn.model_selection import KFold
kf = KFold(n_splits = 5, shuffle = True)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import time

gbt_importance = np.zeros((5,np.shape(measure)[1]))
rf_importance = np.zeros((5,np.shape(measure)[1]))
i = 0
for train_index, test_index in kf.split(measure):
    
    X_train, X_test = measure[train_index], measure[test_index]
    y_train, y_test = label[train_index], label[test_index]
    start = time.time()
    gbt_clf =  GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01, max_features = 'auto',\
                                      max_depth=4, subsample = 0.7).fit(X_train,y_train)
    gbt_importance[i,:] = gbt_clf.feature_importances_
    end = time.time()
    print('gbt running time: ', end-start)        
    print('gbt report: ', classification_report(y_test,gbt_clf.predict(X_test)))
    print('gbt auc: ', roc_auc_score(y_test, gbt_clf.predict_proba(X_test)[:,1]))
    del gbt_clf
    
    start = time.time()
    rf_clf = RandomForestClassifier(n_estimators=1000, max_features = 'auto',\
                                 max_depth=4, class_weight = 'balanced').fit(X_train,y_train)
    rf_importance[i,:] = rf_clf.feature_importances_
    end = time.time()
    print('rf running time: ', end-start)        
    print('rf report: ', classification_report(y_test,rf_clf.predict(X_test)))
    print('rf auc: ', roc_auc_score(y_test, rf_clf.predict_proba(X_test)[:,1]))   
    del rf_clf
    
### feature importance analysis

# gbt_imp = np.mean(gbt_importance,axis = 0)
# rf_imp = np.mean(rf_importance,axis = 0)
# print('gbt feature importance: ', gbt_imp)
# print('rf feature importance: ', rf_imp)

# x = np.linspace(0,35,36)
# import matplotlib.pyplot as plt

# color_list = ['r' for _ in range(12)]
# color_list = color_list + ['b' for _ in range(3)]
# color_list = color_list + ['g' for _ in range(21)]
# barlist = plt.bar(x,gbt_imp,color=color_list)

# barlist[0].set_label('clinical measure')

# barlist[12].set_label('race/gender/age')

# barlist[15].set_label('med')


# plt.legend()
# plt.savefig('gbt_importance.jpg',dpi=200)

# plt.clf()
# barlist = plt.bar(x,rf_imp,color=color_list)

# barlist[0].set_label('clinical measure')

# barlist[12].set_label('race/gender/age')

# barlist[15].set_label('med')


# plt.legend()
# plt.savefig('rf_importance.jpg',dpi=200)