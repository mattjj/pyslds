from __future__ import division
import numpy as np
from numpy.random import randn

X = randn(16,16); X = X.dot(X.T)
A = randn(16,8)

temp = np.zeros_like(A)
out = np.zeros((A.shape[1],A.shape[1]))

import lds
lds.qform2(X,A,temp,out)

print temp
print
print X.dot(A)
print
print
print out
print
print A.T.dot(X).dot(A)
print
print

AT = A.T.copy()
lds.qform1(X,AT,out)

print out

