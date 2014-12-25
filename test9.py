from __future__ import division
import numpy as np
from numpy.random import randn

from states import kf, ks_sample_backwards, kf_resample_lds
from lds import kalman_filter, filter_and_sample

def rand_psd(n):
    A = randn(n,n)
    return A.dot(A.T)

def rand_lds(n,p):
    A = 0.8*np.eye(n)
    B = randn(n,n)
    C = randn(p,n)
    D = rand_psd(p)

    init_mu = np.zeros(n)
    init_sigma = np.eye(n)

    return A, B, C, D, init_mu, init_sigma

if __name__ == '__main__':
    np.seterr(invalid='raise')

    n, p, T = 8, 16, 1000

    A, B, C, D, init_mu, init_sigma = rand_lds(n,p)
    data = np.cumsum(randn(T,p),axis=0)

    BBT = B.dot(B.T)
    DDT = D.dot(D.T)

    filtered_mus, filtered_sigmas = kf(init_mu,init_sigma,[A]*T,[BBT]*T,[C]*T,[DDT]*T,data)
    filtered_mus2, filtered_sigmas2 = kalman_filter(init_mu, init_sigma, A, BBT, C, DDT, data)
    print np.allclose(filtered_mus,filtered_mus2)
    print np.allclose(filtered_sigmas,filtered_sigmas2)

    np.random.seed(0)
    x = ks_sample_backwards([A]*T, [BBT]*T, filtered_mus, filtered_sigmas)
    np.random.seed(0)
    x2 = filter_and_sample(init_mu, init_sigma, A, BBT, C, DDT, data)
    print np.allclose(x,x2)

