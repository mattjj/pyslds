from __future__ import division
import numpy as np
from pylab import *

from states import condition_on_python
from util import condition_on

def rand_psd(n):
    out = randn(n,2*n)
    return out.dot(out.T)

mu_x = randn(3)
sigma_x = rand_psd(3)
A = randn(5,3)
sigma_obs = rand_psd(5)
y = randn(5)

mu_out_py, sigma_out_py = condition_on_python(mu_x,sigma_x,A,sigma_obs,y)

mu_out, sigma_out = np.empty_like(mu_x), np.empty_like(sigma_x)
condition_on(mu_x,sigma_x,A,sigma_obs,y,mu_out,sigma_out)

print np.allclose(mu_out,mu_out_py)
print np.allclose(sigma_out,sigma_out_py)

