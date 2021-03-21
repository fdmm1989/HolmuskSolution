# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 21:24:11 2021

@author: MengZY
"""
import numpy as np
med = [[] for _ in range(300)]

f = open('T_meds.csv')

f.readline()
for l in f:
    i = int(l.split(',')[0])
    med[i].append(l.split(',')[1])

f = open('T_stage.csv')
f.readline()

import matplotlib.pyplot as plt

meds = {'irbesartan':0, 'dapagliflozin':1, 'lovastatin':2, 'rosuvastatin':3, \
        'nebivolol':4, 'valsartan':5, 'bisoprolol':6, 'pravastatin':7, \
        'metoprolol':8, 'olmesartan':9, 'simvastatin':10, 'pitavastatin':11, \
        'losartan':12, 'metformin':13, 'atenolol':14, 'atorvastatin':15, \
        'canagliflozin':16, 'carvedilol':17, 'telmisartan':18, 'labetalol':19, \
        'propranolol':20}

plt.clf()
med_stat = np.zeros((21,2))


for l in f:
    i = int(l.split(',')[0])
    if l.split(',')[1] == 'False\n':
        for m in meds:
            if m in med[i]:
                med_stat[meds[m],1] += 1
    else:
        for m in meds:
            if m in med[i]:
                med_stat[meds[m],0] += 1

x = np.linspace(0,20,21)
plt.plot(x,med_stat[:,0],'.-b',label= 'True')
plt.plot(x,med_stat[:,1],'.-r',label= 'False')
plt.legend()
plt.show()
        