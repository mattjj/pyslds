from __future__ import division
import numpy as np
from numpy.random import randn

m, n, k = np.random.randint(2,10,size=3)

A = randn(m,k)
B = randn(k,n)
x = randn(k)

vout = np.zeros(A.shape[0])
mout = np.zeros((A.shape[0],B.shape[1]))

from lds import foo
foo(A,x,vout)
print np.allclose(vout, A.dot(x))

from lds import foo2
foo2(A,B,mout)
print np.allclose(mout, A.dot(B))


A = A.dot(A.T)
B = randn(A.shape[1],n)
x = randn(A.shape[1])

from lds import foo3
foo3(A,x,vout)
print np.allclose(vout, A.dot(x))

from lds import foo4
foo4(A,B,mout)
print np.allclose(mout, A.dot(B))

