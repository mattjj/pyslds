from __future__ import division
import numpy as np
from numpy.random import randn

n = 8
p = 16

sigma_x = randn(n,n)
sigma_x = np.asfortranarray(sigma_x.dot(sigma_x.T))

mu_x = np.asfortranarray(randn(n))

y = np.asfortranarray(randn(p))

C = np.asfortranarray(randn(p,n))
A = np.asfortranarray(randn(n,n))

sigma_states = np.asfortranarray(0.5*np.eye(n))
sigma_obs = np.asfortranarray(np.eye(p))


mu_cond = np.asfortranarray(np.zeros_like(mu_x))
sigma_cond = np.asfortranarray(np.zeros_like(sigma_x))

mu_predict = np.asfortranarray(np.zeros_like(mu_x))
sigma_predict = np.asfortranarray(np.zeros_like(sigma_x))


from lds import condition_on
condition_on(mu_x, sigma_x, A, sigma_states, C, sigma_obs, y, mu_cond, sigma_cond, mu_predict, sigma_predict)


sigma_y = C.dot(sigma_x).dot(C.T) + sigma_obs

print (sigma_x.dot(C.T).dot(np.linalg.solve(sigma_y, y - C.dot(mu_x))) + mu_x)[:3]
print mu_cond[:3]
print
print np.tril(sigma_cond)[:3,:3]
print np.tril(sigma_x - sigma_x.dot(C.T).dot(np.linalg.solve(sigma_y,C.dot(sigma_x))))[:3,:3]
print
print mu_predict[:3]
print A.dot(mu_cond)[:3]
print
print sigma_predict[:3,:3]
print (A.dot(sigma_x - sigma_x.dot(C.T).dot(np.linalg.solve(sigma_y,C.dot(sigma_x)))).dot(A.T) + sigma_states)[:3,:3]

from lds2 import condition_on as condition_on2
condition_on2(*map(np.ascontiguousarray, [mu_x, sigma_x, A, sigma_obs, y, mu_cond, sigma_cond]))


