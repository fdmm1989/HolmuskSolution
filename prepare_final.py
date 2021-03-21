# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 19:44:32 2021

@author: MengZY
"""
import numpy as np
# measure
measure_list = ['T_creatinine.csv', 'T_DBP.csv', 'T_glucose.csv', 'T_HGB.csv', 'T_ldl.csv', 'T_SBP.csv']
N = 300
feature = np.zeros((N,6*2+1))
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

# age

f = open('T_demo.csv')
f.readline()
x = 0
for l in f:
    i = int(l.split(',')[0])
    a = int(l.split(',')[3])
    feature[x, 12] = a
    x += 1

f.close()


            
            
np.save('X_final.npy',feature)
