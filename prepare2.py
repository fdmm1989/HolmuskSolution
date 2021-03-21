# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 16:14:38 2021

@author: MengZY
"""

import numpy as np
# measure
measure_list = ['T_creatinine.csv', 'T_DBP.csv', 'T_glucose.csv', 'T_HGB.csv', 'T_ldl.csv', 'T_SBP.csv']
N = 300
feature = np.zeros((N,6*2+3+21))
m = 0
for measure in measure_list:
    f = open(measure)
    f.readline()
    data = []
    for l in f:
        i = int(l.split(',')[0])
        v = float(l.split(',')[1])
        data.append([i,v])
    f.close()
    # max, min, avg
    data = np.array(data)
    for x in range(N):
        data1 = data[data[:,0]==x,1]
        feature[x,m*2 + 0] = data1[0]
        feature[x,m*2 + 1] = data1[-1]
        
    m += 1

# race, gender, age
race = {'Unknown':0, 'White':1, 'Black':2, 'Asian':3, 'Hispanic':4}
gender = {'Male':0, 'Female':1}
f = open('T_demo.csv')
f.readline()
x = 0
for l in f:
    i = int(l.split(',')[0])
    r = l.split(',')[1]
    s = l.split(',')[2]
    a = int(l.split(',')[3])
    feature[x, 12] = race[r]
    feature[x, 13] = gender[s]
    feature[x, 14] = a
    x += 1

f.close()

# med
meds = {'irbesartan':0, 'dapagliflozin':1, 'lovastatin':2, 'rosuvastatin':3, \
        'nebivolol':4, 'valsartan':5, 'bisoprolol':6, 'pravastatin':7, \
        'metoprolol':8, 'olmesartan':9, 'simvastatin':10, 'pitavastatin':11, \
        'losartan':12, 'metformin':13, 'atenolol':14, 'atorvastatin':15, \
        'canagliflozin':16, 'carvedilol':17, 'telmisartan':18, 'labetalol':19, \
        'propranolol':20}

f = open('T_meds.csv')
f.readline()
data = []
for l in f:
    i = int(l.split(',')[0])
    m = meds[l.split(',')[1]]
    d = float(l.split(',')[2])
    s = int(l.split(',')[3])
    e = int(l.split(',')[4])
    data.append([i,m,d,s,e])

f.close()
data = np.array(data)

for x in range(N):

    data1 = data[data[:,0]==x,1:5]
    if len(data1) == 0:        
        continue
        
    total_time = np.zeros(len(meds))
    for d in data1:        
        feature[x, 15 + int(d[0])] += d[1]*(d[3]-d[2])
        total_time[int(d[0])] += (d[3]-d[2])

    for i in range(len(meds)):
        if total_time[i] > 0:
            feature[x, 15 + i] = feature[x, 15 + i] / total_time[i]
            
            
np.save('X2.npy',feature)