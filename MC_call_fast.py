# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 12:38:16 2016

@author: Yupeng

Price Vanilla call with GBM by MC
"""

import numpy as np
import numpy.random as npr
import time

S0 = 100.0
K = 100.0
r = 0.05
vol = 0.5
T = 1.0

"""""""""""""""""""""""""""""""""
faster method : vectorisation
"""""""""""""""""""""""""""""""""

t_fast_begin = time.time()
I = 50000
M = 10
S = np.zeros((M, I))
phi = npr.standard_normal((M, I))
S[0] = 100 * np.ones((I,))
dt = T/M
for i in range(M - 1):
    S[i+1] = S[i] * (1 + r * dt + vol * np.sqrt(dt) * phi[i])
payoff = np.maximum(S[M-1] - K, 0)
value_call = np.mean(payoff * np.exp(-r * T))
std_call = np.std(payoff * np.exp(-r * T)) / np.sqrt(I)
print('Call price is %f' %value_call)
print('The error is %f' %std_call)
t_fast_end = time.time()
t_fast = -t_fast_begin + t_fast_end
print('Time consumption is %f seconds' %t_fast)


"""""""""""""""""""""""""""""""""
slower method : by loop
"""""""""""""""""""""""""""""""""

t_slow_begin = time.time()
I = 50000
M = 10
S = np.zeros((M, I))
phi = npr.standard_normal((M, I))
S[0] = 100 * np.ones((I,))
dt = T/M
for i in range(M - 1):
    for j in range(I):
        S[i+1, j] = S[i, j] * (1 + r * dt + vol * np.sqrt(dt) * phi[i, j])
payoff = np.maximum(S[M-1] - K, 0)
value_call = np.mean(payoff * np.exp(-r * T))
print('Call price is %f' %value_call)
t_slow_end = time.time()
t_slow = -t_slow_begin + t_slow_end
print('Time consumption is %f seconds' %t_slow)


"""""""""""""""""""""""""""""""""
faster method : knock out call
"""""""""""""""""""""""""""""""""

t_fast_begin = time.time()
I = 50000
M = 10
barrier = 150
S = np.zeros((M, I))
phi = npr.standard_normal((M, I))
S[0] = 100 * np.ones((I,))
dt = T/M
for i in range(M - 1):
    S[i+1] = S[i] * (1 + r * dt + vol * np.sqrt(dt) * phi[i])
mat_barrier = S - barrier
mat_indicator = np.int64(mat_barrier < 0)
indicator = np.prod(mat_indicator, axis = 0)
payoff = np.maximum(S[M-1] - K, 0) * indicator
value_call = np.mean(payoff * np.exp(-r * T))
std_call = np.std(payoff * np.exp(-r * T)) / np.sqrt(I)
print('Barrier-Call price is %f' %value_call)
print('The error is %f' %std_call)
t_fast_end = time.time()
t_fast = -t_fast_begin + t_fast_end
print('Time consumption is %f seconds' %t_fast)
