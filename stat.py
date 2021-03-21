# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 16:06:05 2021

@author: MengZY
"""


value = [[] for _ in range(300)]
time = [[] for _ in range(300)]
f = open('T_meds.csv')

f.readline()
for l in f:
 
    i = int(l.split(',')[0])
    value[i].append(float(l.split(',')[1]))
    time[i].append(float(l.split(',')[2]))
    
f = open('T_stage.csv')
f.readline()

import matplotlib.pyplot as plt

plt.clf()

for l in f:
    i = int(l.split(',')[0])
    if l.split(',')[1] == 'False\n':
        l1, = plt.plot(time[i],value[i],'-b',label='False')
        
    else:
        l2, = plt.plot(time[i],value[i],'-r',label='True')

plt.legend(handles = [l1,l2])
plt.show()




