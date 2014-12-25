# distutils: extra_compile_args = -O2 -w
# cython: boundscheck = False
# cython: nonecheck = False
# cython: wraparound = False
# cython: cdivision = True

import numpy as np
cimport numpy as np
cimport cython

from blas_lapack cimport r_gemv, r_gemm, r_symv, r_symm, \
        matrix_qform, chol_update, chol_downdate
from blas_lapack cimport dsymm, dcopy, dgemm, dpotrf, \
        dgemv, dpotrs, daxpy, dtrtrs, dsyrk, dtrmv

# TODO make switching version
# TODO make cholesky update/downdate versions
# TODO make (generate?) single-precision version

# NOTE: I tried the dsymm / dsyrk version, and it was slower! (on my laptop)
# but is it more stable? basic tests suggest that's unimportant. symmetric
# routines commented out inline below.
# NOTE: for symmetric matrices, F/C order doesn't matter

def kalman_filter(
    double[::1] mu_init, double[:,:] sigma_init,
    double[:,:] A, double[:,:] sigma_states,
    double[:,:] C, double[:,:] sigma_obs,
    double[:,::1] data):

    # allocate temporaries and internals
    cdef int t, T = data.shape[0]
    cdef int n = C.shape[1], p = C.shape[0]
    cdef int nn = n*n, pp = p*p
    cdef double one = 1., zero = 0., neg1 = -1.
    cdef int inc = 1, info = 0

    cdef double[::1] mu_predict = np.empty(n)
    cdef double[:,:] sigma_predict = np.empty((n,n),order='F')

    cdef double[::1,:] temp1 = np.empty((p,p),order='F')
    cdef double[::1,:] temp2 = np.empty((p,n),order='F')
    cdef double[::1]   temp3 = np.empty((p,), order='F')
    cdef double[::1,:] temp4 = np.empty((n,n),order='F')

    # allocate output
    cdef double[:,::1] filtered_mus = np.empty((T,n))
    cdef double[:,:,::1] filtered_sigmas = np.empty((T,n,n))

    # make inputs contiguous and fortran-order
    A, sigma_states = np.asfortranarray(A), np.asfortranarray(sigma_states)
    C, sigma_obs = np.asfortranarray(C), np.asfortranarray(sigma_obs)

    # run filter forwards
    mu_predict[...] = mu_init
    sigma_predict[...] = sigma_init
    for t in range(T):
        ### temp1 = chol(C * sigma_predict * C' + sigma_obs)
        dgemm('N', 'N', &p, &n, &n, &one, &C[0,0], &p, &sigma_predict[0,0], &n, &zero, &temp2[0,0], &p)
        # dsymm('R','L', &p, &n, &one, &sigma_predict[0,0], &n, &C[0,0], &p, &zero, &temp2[0,0], &p)
        dcopy(&pp, &sigma_obs[0,0], &inc, &temp1[0,0], &inc)
        dgemm('N', 'T', &p, &p, &n, &one, &temp2[0,0], &p, &C[0,0], &p, &one, &temp1[0,0], &p)
        dpotrf('L', &p, &temp1[0,0], &p, &info)

        ### filtered_mus[t] = mu_predict + sigma_predict * C' * inv_from_chol(temp1) * (y - C * mu_predict)
        dcopy(&p, &data[t,0], &inc, &temp3[0], &inc)
        dgemv('N', &p, &n, &neg1, &C[0,0], &p, &mu_predict[0], &inc, &one, &temp3[0], &inc)
        dpotrs('L', &p, &inc, &temp1[0,0], &p, &temp3[0], &p, &info)
        dcopy(&n, &mu_predict[0], &inc, &filtered_mus[t,0], &inc)
        dgemv('T', &p, &n, &one, &temp2[0,0], &p, &temp3[0], &inc, &one, &filtered_mus[t,0], &inc)

        ### filtered_sigmas[t] = sigma_predict - solve(temp1, C * sigma_x)' * solve(temp1, C * sigma_x)
        dtrtrs('L', 'N', 'N', &p, &n, &temp1[0,0], &p, &temp2[0,0], &p, &info)
        dcopy(&nn, &sigma_predict[0,0], &inc, &filtered_sigmas[t,0,0], &inc)
        dgemm('T', 'N', &n, &n, &p, &neg1, &temp2[0,0], &p, &temp2[0,0], &p, &one, &filtered_sigmas[t,0,0], &n)
        # dsyrk('L','T', &n, &p, &neg1, &temp2[0,0], &p, &one, &filtered_sigmas[t,0,0], &n)

        ### mu_predict = A * filtered_mus[t]
        dgemv('N', &n, &n, &one, &A[0,0], &n, &filtered_mus[t,0], &inc, &zero, &mu_predict[0], &inc)

        ### sigma_predict = A * filtered_sigmas[t] * A' + sigma_states
        dgemm('N', 'N', &n, &n, &n, &one, &A[0,0], &n, &filtered_sigmas[t,0,0], &n, &zero, &temp4[0,0], &n)
        # dsymm('R','L',&n, &n, &one, &filtered_sigmas[t,0,0], &n, &A[0,0], &n, &zero, &temp4[0,0], &n)
        dcopy(&nn, &sigma_states[0,0], &inc, &sigma_predict[0,0], &inc)
        dgemm('N', 'T', &n, &n, &n, &one, &temp4[0,0], &n, &A[0,0], &n, &one, &sigma_predict[0,0], &n)

    return np.asarray(filtered_mus), np.asarray(filtered_sigmas)

def filter_and_sample(
    double[::1] mu_init, double[:,:] sigma_init,
    double[:,:] A, double[:,:] sigma_states,
    double[:,:] C, double[:,:] sigma_obs,
    double[:,::1] data):

    ### allocate temporaries and internals
    cdef int t, T = data.shape[0]
    cdef int n = C.shape[1], p = C.shape[0]
    cdef int nn = n*n, pp = p*p
    cdef double one = 1., zero = 0., neg1 = -1.
    cdef int inc = 1, info = 0

    cdef double[::1] mu_predict = np.empty(n)
    cdef double[:,:] sigma_predict = np.empty((n,n),order='F')

    cdef double[::1,:] temp1 = np.empty((p,p),order='F')
    cdef double[::1,:] temp2 = np.empty((p,n),order='F')
    cdef double[::1]   temp3 = np.empty((p,), order='F')
    cdef double[::1,:] temp4 = np.empty((n,n),order='F')
    cdef double[::1]   temp5 = np.empty((n,), order='F')

    cdef double[:,::1] filtered_mus = np.empty((T,n))
    cdef double[:,:,::1] filtered_sigmas = np.empty((T,n,n))

    ### allocate output and generate randomness
    cdef double[:,::1] randseq = np.random.randn(T,n)

    ### make inputs contiguous and fortran-order
    A, sigma_states = np.asfortranarray(A), np.asfortranarray(sigma_states)
    C, sigma_obs = np.asfortranarray(C), np.asfortranarray(sigma_obs)

    ### run filter forwards
    mu_predict[...] = mu_init
    sigma_predict[...] = sigma_init
    for t in range(T):
        ### temp1 = chol(C * sigma_predict * C' + sigma_obs)
        dgemm('N', 'N', &p, &n, &n, &one, &C[0,0], &p, &sigma_predict[0,0], &n, &zero, &temp2[0,0], &p)
        # dsymm('R','L', &p, &n, &one, &sigma_predict[0,0], &n, &C[0,0], &p, &zero, &temp2[0,0], &p)
        dcopy(&pp, &sigma_obs[0,0], &inc, &temp1[0,0], &inc)
        dgemm('N', 'T', &p, &p, &n, &one, &temp2[0,0], &p, &C[0,0], &p, &one, &temp1[0,0], &p)
        dpotrf('L', &p, &temp1[0,0], &p, &info)

        ### filtered_mus[t] = mu_predict + sigma_predict * C' * inv_from_chol(temp1) * (y - C * mu_predict)
        dcopy(&p, &data[t,0], &inc, &temp3[0], &inc)
        dgemv('N', &p, &n, &neg1, &C[0,0], &p, &mu_predict[0], &inc, &one, &temp3[0], &inc)
        dpotrs('L', &p, &inc, &temp1[0,0], &p, &temp3[0], &p, &info)
        dcopy(&n, &mu_predict[0], &inc, &filtered_mus[t,0], &inc)
        dgemv('T', &p, &n, &one, &temp2[0,0], &p, &temp3[0], &inc, &one, &filtered_mus[t,0], &inc)

        ### filtered_sigmas[t] = sigma_predict - solve(temp1, C * sigma_x)' * solve(temp1, C * sigma_x)
        dtrtrs('L', 'N', 'N', &p, &n, &temp1[0,0], &p, &temp2[0,0], &p, &info)
        dcopy(&nn, &sigma_predict[0,0], &inc, &filtered_sigmas[t,0,0], &inc)
        dgemm('T', 'N', &n, &n, &p, &neg1, &temp2[0,0], &p, &temp2[0,0], &p, &one, &filtered_sigmas[t,0,0], &n)
        # dsyrk('L','T', &n, &p, &neg1, &temp2[0,0], &p, &one, &filtered_sigmas[t,0,0], &n)

        ### mu_predict = A * filtered_mus[t]
        dgemv('N', &n, &n, &one, &A[0,0], &n, &filtered_mus[t,0], &inc, &zero, &mu_predict[0], &inc)

        ### sigma_predict = A * filtered_sigmas[t] * A' + sigma_states
        dgemm('N', 'N', &n, &n, &n, &one, &A[0,0], &n, &filtered_sigmas[t,0,0], &n, &zero, &temp4[0,0], &n)
        # dsymm('R','L',&n, &n, &one, &filtered_sigmas[t,0,0], &n, &A[0,0], &n, &zero, &temp4[0,0], &n)
        dcopy(&nn, &sigma_states[0,0], &inc, &sigma_predict[0,0], &inc)
        dgemm('N', 'T', &n, &n, &n, &one, &temp4[0,0], &n, &A[0,0], &n, &one, &sigma_predict[0,0], &n)

    ### randseq[T-1] = sample_multivariate_normal(filtered_mus[T-1], filtered_sigmas[T-1])
    dpotrf('L', &n, &filtered_sigmas[T-1,0,0], &n, &info)
    dtrmv('L', 'N', 'N', &n, &filtered_sigmas[T-1,0,0], &n, &randseq[T-1,0], &inc)
    daxpy(&n, &one, &filtered_mus[T-1,0], &inc, &randseq[T-1,0], &inc)
    for t in range(T-2,-1,-1):
        ### sigma_predict = chol(A * filtered_sigmas[t,0,0] * A' + sigma_states)
        dgemm('N', 'N', &n, &n, &n, &one, &A[0,0], &n, &filtered_sigmas[t,0,0], &n, &zero, &temp4[0,0], &n)
        dcopy(&nn, &sigma_states[0,0], &inc, &sigma_predict[0,0], &inc)
        dgemm('N', 'T', &n, &n, &n, &one, &temp4[0,0], &n, &A[0,0], &n, &one, &sigma_predict[0,0], &n)
        dpotrf('L', &n, &sigma_predict[0,0], &n, &info)

        ### filtered_mus[t] += filtered_sigmas[t] * A' * inv_from_chol(sigma_predict) \
        ###                     * (randseq[t+1] - A * filtered_mus[t])
        dcopy(&n, &randseq[t+1,0], &inc, &temp5[0], &inc)
        dgemv('N', &n, &n, &neg1, &A[0,0], &n, &filtered_mus[t,0], &inc, &one, &temp5[0], &inc)
        dpotrs('L', &n, &inc, &sigma_predict[0,0], &n, &temp5[0], &n, &info)
        dgemv('T', &n, &n, &one, &temp4[0,0], &n, &temp5[0], &inc, &one, &filtered_mus[t,0], &inc)

        ### filtered_sigmas[t] -= solve(sigma_predict, A * filtered_sigmas[t])'
        ###                             * solve(sigma_predict, A*filtered_sigmas[t])
        dtrtrs('L', 'N', 'N', &n, &n, &sigma_predict[0,0], &n, &temp4[0,0], &n, &info)
        dgemm('T', 'N', &n, &n, &n, &neg1, &temp4[0,0], &n, &temp4[0,0], &n, &one, &filtered_sigmas[t,0,0], &n)

        ### randseq[t] = sample_multivariate_normal(filtered_mus[t], filtered_sigmas[t])
        dpotrf('L', &n, &filtered_sigmas[t,0,0], &n, &info)
        dtrmv('L', 'N', 'N', &n, &filtered_sigmas[t,0,0], &n, &randseq[t,0], &inc)
        daxpy(&n, &one, &filtered_mus[t,0], &inc, &randseq[t,0], &inc)

    return np.asarray(randseq)

###

def condition_on(
    # inputs
    double[::1] mu_x, double[::1,:] sigma_x,
    double[::1,:] A, double[::1,:] sigma_states,
    double[::1,:] C, double[::1,:] sigma_obs,
    double[::1] y,
    # outputs
    double[::1] mu_cond, double[::1,:] sigma_cond,
    double[::1] mu_predict, double[::1,:] sigma_predict,
    ):
    cdef int n = mu_x.shape[0]
    cdef int p = C.shape[0]
    cdef int nn = n*n, pp = p*p
    cdef double one = 1., zero = 0., neg1 = -1.
    cdef int inc = 1, info = 0

    cdef double[::1,:] sigma_y = np.empty((p,p),order='F')
    cdef double[::1,:] temp1 = np.empty((p,n),order='F')
    cdef double[::1] temp2 = np.empty((p,),order='F')
    cdef double[::1,:] temp4 = np.empty((n,n),order='F')

    ### sigma_y = chol(sigma_obs + C.dot(sigma_x).dot(C.T))
    ## temp1 = C.dot(sigma_x)
    dgemm('N', 'N', &p, &n, &n, &one, &C[0,0], &p, &sigma_x[0,0], &n, &zero, &temp1[0,0], &p)
    # dsymm('R','L', &p, &n, &one, &sigma_x[0,0], &n, &C[0,0], &p, &zero, &temp1[0,0], &p)
    ## sigma_y = sigma_obs
    dcopy(&pp, &sigma_obs[0,0], &inc, &sigma_y[0,0], &inc)
    ## sigma_y += temp1.dot(C.T)
    dgemm('N', 'T', &p, &p, &n, &one, &temp1[0,0], &p, &C[0,0], &p, &one, &sigma_y[0,0], &p)
    ## sigma_y = chol(sigma_y)
    dpotrf('L', &p, &sigma_y[0,0], &p, &info)

    ### temp2 = y - C * mu_x
    ## temp2 = y
    dcopy(&p, &y[0], &inc, &temp2[0], &inc)
    ## temp2 -= C * mu_x
    dgemv('N', &p, &n, &neg1, &C[0,0], &p, &mu_x[0], &inc, &one, &temp2[0], &inc)

    ### mu_cond = mu_x + temp1' * solve_from_chol(sigma_y, temp2)
    ## temp2 = solve(sigma_y, temp2)
    dpotrs('L', &p, &inc, &sigma_y[0,0], &p, &temp2[0], &p, &info)
    ## mu_cond = mu_x
    dcopy(&n, &mu_x[0], &inc, &mu_cond[0], &inc)
    ## mu_cond += temp1.dot(temp2)
    dgemv('T', &p, &n, &one, &temp1[0,0], &p, &temp2[0], &inc, &one, &mu_cond[0], &inc)

    ### sigma_cond = sigma_x - temp1.T.dot(sigma_y^{-1}).temp1
    ## temp1 = solve_triangular(sigma_y, temp1)
    dtrtrs('L', 'N', 'N', &p, &n, &sigma_y[0,0], &p, &temp1[0,0], &p, &info)
    ## sigma_cond = sigma_x
    dcopy(&nn, &sigma_x[0,0], &inc, &sigma_cond[0,0], &inc)
    ## sigma_cond -= temp1.T.dot(temp1)
    dgemm('T', 'N', &n, &n, &p, &neg1, &temp1[0,0], &p, &temp1[0,0], &p, &one, &sigma_cond[0,0], &n)
    # dsyrk('L','T', &n, &p, &neg1, &temp1[0,0], &p, &one, &sigma_cond[0,0], &n)

    ### mu_predict = A.dot(mu_cond)
    dgemv('N', &n, &n, &one, &A[0,0], &n, &mu_cond[0], &inc, &zero, &mu_predict[0], &inc)

    ### sigma_predict = A.dot(sigma_cond).dot(A.T) + sigma_states
    ## temp4 = A.dot(sigma_cond)
    dgemm('N', 'N', &n, &n, &n, &one, &A[0,0], &n, &sigma_cond[0,0], &n, &zero, &temp4[0,0], &n)
    # dsymm('R','L',&n, &n, &one, &sigma_cond[0,0], &n, &A[0,0], &n, &zero, &temp4[0,0], &n)
    # sigma_predict = sigma_states
    dcopy(&nn, &sigma_states[0,0], &inc, &sigma_predict[0,0], &inc)
    # sigma_predict += temp4.dot(A.T)
    dgemm('N', 'T', &n, &n, &n, &one, &temp4[0,0], &n, &A[0,0], &n, &one, &sigma_predict[0,0], &n)

### TEST

def test_dgemm(double[::1,:] A, double[::1,:] B, double[::1,:] C):
    cdef int i, m = A.shape[0], n = B.shape[1], k = A.shape[1]
    cdef double one = 1., zero = 0.

    for i in range(1000):
        dgemm('N','N', &m, &n, &k,
                &one, &A[0,0], &m, &B[0,0], &k,
                &zero, &C[0,0], &m)

def test_dsymm(double[::1,:] A, double[::1,:] B, double[::1,:] C):
    cdef int i, m = A.shape[0], n = B.shape[0]
    cdef double one = 1., zero = 0.

    for i in range(1000):
        dsymm('L','L', &m, &n,
                &one, &A[0,0], &m, &B[0,0], &m,
                &zero, &C[0,0], &m)


def foo(double[:,::1] A, double[::1] x, double[::1] out):
    r_gemv(A,x,out)

def foo2(double[:,::1] A, double[:,::1] B, double[:,::1] out):
    r_gemm(A,B,out)

def foo3(double[:,::1] A, double[::1] x, double[::1] out):
    r_symv("L",A,x,out)

def foo4(double[:,::1] A, double[:,::1] B, double[:,::1] out):
    r_symm("L","L",A,B,out)


def qform1(double[:,::1] X, double[:,::1] A, double[:,::1] out):
    cdef int i
    for i in range(1000):
        matrix_qform(A.shape[0], A.shape[1], &A[0,0], &X[0,0], 0., &out[0,0])

# TODO whoops A is assumed to be coming in transposed here!
def qform2(double[:,::1] X, double[:,::1] A, double[:,::1] temp, double[:,::1] out):
    cdef int j
    cdef double alpha = 1., beta = 0.
    cdef int inc = 1

    # NOTE: tricky because row-major; everything is flipped!

    for j in range(1000):
        dsymm('R', 'L', <int*> &A.shape[1], <int*> &A.shape[0],
                &alpha, &X[0,0], <int*> &X.shape[1],
                &A[0,0], <int*> &A.shape[1],
                &beta, &temp[0,0], <int*> &temp.shape[1])

        dgemm('N', 'T', <int*> &out.shape[0], <int*> &out.shape[0], <int*> &X.shape[0],
                &alpha, &A[0,0], <int*> &A.shape[1],
                &temp[0,0], <int*> &temp.shape[1],
                &beta, &out[0,0], <int*> &out.shape[0])

# NOTE: qform2 is much faster! 1.5 vs 9.75. maybe because it's reading both sides?


def downdate(double[:,::1] R, double[::1] z):
    chol_downdate(R.shape[0],&R[0,0],&z[0])
    return np.asarray(R)

def update(double[:,::1] R, double[::1] z):
    chol_update(R.shape[0],&R[0,0],&z[0])
    return np.asarray(R)


def chol_downdate_rankk(double[:,::1] R, double[:,::1] Z):
    cdef int i

    cdef int j
    for j in range(1000):

        for i in range(Z.shape[0]):
            chol_downdate(R.shape[0],&R[0,0],&Z[i,0])

