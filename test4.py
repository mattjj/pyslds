from __future__ import division
import numpy as np
from numpy.random import randn

n = 50
k = 10

A = randn(n,5*n)
A = A.dot(A.T)

R = np.linalg.cholesky(A).T.copy()
Z = randn(k,n)
Z = np.zeros_like(Z)

import lds
lds.chol_downdate_rankk(R,Z.copy())

R2 = np.linalg.cholesky(A - Z.T.dot(Z)).T

print np.allclose(R, R2)

