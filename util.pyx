# distutils: extra_compile_args = -O3 -w -DNDEBUG -std=c++11 -I/Users/mattjj/Desktop/pyhsmm/internals/ -I/Users/mattjj/Desktop/pyhsmm/deps/Eigen3/

import numpy as np
cimport numpy as np

from libc.stdint cimport int32_t
from cython cimport floating, integral

from cython.parallel import prange

# TODO this takes FOREVER to compile with cython.floating

cdef extern from "slds_util.h":
    cdef cppclass dummy[Type]:
        dummy()
        void condition_on(
            int D, int P,
            Type *mu_x, Type *sigma_x, Type *A, Type *sigma_obs, Type *y,
            Type *mu_out, Type *sigma_out) nogil

        # void kf_resample_lds(
        #     int T, int D, int P,
        #     Type *As, Type *BBTs, Type *Cs, Type *DDTs,
        #     Type *data, Type *randseq, Type *out) nogil

def condition_on(
        double[::1] mu_x,
        double[:,::1] sigma_x,
        double[:,::1] A,
        double[:,::1] sigma_obs,
        double[::1] y,
        double[::1] mu_out,
        double[:,::1] sigma_out):
    cdef dummy[double] ref
    ref.condition_on(y.shape[0],mu_x.shape[0],&mu_x[0],&sigma_x[0,0],
            &A[0,0],&sigma_obs[0,0],&y[0],&mu_out[0],&sigma_out[0,0])

# # TODO this function should accept lists of these objects so we can parallelize
# def kf_resample_lds(
#         floating[::1] init_mu,
#         floating[:,::1] init_sigma,
#         floating[:,:,::1] As,
#         floating[:,:,::1] BBTs,
#         floating[:,:,::1] Cs,
#         floating[:,:,::1] DDTs,
#         floating[:,::1] emissions,
#         floating[:,::1] out):
#     cdef dummy[floating] ref
#     cdef int T = emissions.shape[0]
#     cdef int D = emissions.shape[1]
#     cdef int P = out.shape[1]

#     cdef floating[:,::1]
#     if floating is double:
#         randseq = np.random.normal(size=((T,P))).astype(np.double)
#     else:
#         randseq = np.random.normal(size=((T,P))).astype(np.float)

#     ref.kf_resample_lds(
#             T,D,P,
#             &As[0,0,0],&BBTs[0,0,0],&Cs[0,0,0],&DDTs[0,0,0],
#             &emissions[0,0],&randseq[0,0],&out[0,0])

