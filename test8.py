from __future__ import division
import numpy as np
from numpy.random import randn

A = randn(12,12)
A = A.dot(A.T)
A = np.asfortranarray(A)

B = np.asfortranarray(randn(12,16))

C = np.asfortranarray(np.zeros((A.shape[0],B.shape[1])))

from lds import test_dgemm, test_dsymm

