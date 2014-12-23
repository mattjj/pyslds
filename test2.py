from __future__ import division
import numpy as np
from numpy.random import randn

m,k = 4,2

X = randn(k,k); X = X.dot(X.T)
A = randn(m,k)
out = np.zeros((m,m))

import lds
lds.bar(X,A,out)

print out

