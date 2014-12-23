# distutils: extra_compile_args = -O3 -w -ffast-math
# cython: boundscheck = False
# cython: nonecheck = False
# cython: wraparound = False
# cython: cdivision = True

from cython cimport floating
from libc.math cimport sqrt

##########
#  BLAS  #
##########

# http://www.netlib.org/blas/

ctypedef int dcopy_t(
    int *n, double *dx, int *incx, double *dy, int *incy
    ) nogil
cdef dcopy_t *dcopy

ctypedef int daxpy_t(
    int *n, double *da, double *dx, int *incx, double *dy, int *incy
    ) nogil
cdef daxpy_t *daxpy

ctypedef int drotg_t(
    double *da, double *db, double *c ,double *s
    ) nogil
cdef drotg_t *drotg

ctypedef int drot_t(
    int *n, double *dx, int *incx, double *dy, int *incy, double *c, double *s
    ) nogil
cdef drot_t *drot

ctypedef int srotg_t(
    float *da, float *db, float *c ,float *s
    ) nogil
cdef srotg_t *srotg

ctypedef int srot_t(
    int *n, float *dx, int *incx, float *dy, int *incy, float *c, float *s
    ) nogil
cdef srot_t *srot

ctypedef double ddot_t(
    int *n, double *x, int *incx, double *y, int *incy
    ) nogil
cdef ddot_t *ddot

ctypedef int dgemv_t(
    char *trans,
    int *m, int *n, double *alpha, double *A, int *lda,
    double *x, int *incx,
    double *beta, double *y, int *incy
    ) nogil
cdef dgemv_t *dgemv

ctypedef int dgemm_t(
    char *transa, char *transb, int *m, int *n, int *k,
    double *alpha, double *A, int *lda,
    double *B, int *ldb,
    double *beta, double *C, int *ldc
    ) nogil
cdef dgemm_t *dgemm

ctypedef int dsymv_t(
    char *uplo, int *n,
    double *alpha, double *A, int *lda,
    double *x, int *incx,
    double *beta, double *y, int *incy
    ) nogil
cdef dsymv_t *dsymv

ctypedef int dsymm_t(
    char *side, char *uplo, int *m, int *n,
    double *alpha, double *A, int *lda,
    double *B, int *ldb,
    double *beta, double *C, int *ldc
    ) nogil
cdef dsymm_t *dsymm

ctypedef double dger_t(
    int *m, int *n, double *alpha,
    double *x, int *incx,
    double *y, int *incy,
    double *A, int *lda
    ) nogil
cdef dger_t *dger

############
#  LAPACK  #
############

# http://www.netlib.org/lapack/

ctypedef int dposv_t(
    char *uplo, int *n, int *nrhs,
    double *A, int *lda,
    double *B, int *ldb,
    int *info) nogil
cdef dposv_t *dposv

ctypedef int dpotrf_t(
    char *uplo, int *n,
    double *a, int *lda,
    int *info) nogil
cdef dpotrf_t *dpotrf

ctypedef int dpotrs_t(
    char *uplo, int *n, int *nrhs,
    double *a, int *lda,
    double *b, int *ldb,
    int *info) nogil
cdef dpotrs_t *dpotrs

ctypedef int dtrtrs_t(
    char *uplo, char *trans, char *diag, int *n, int *nrhs,
    double *A, int *lda,
    double *B, int *ldb,
    int *info) nogil
cdef dtrtrs_t *dtrtrs

ctypedef int dsyrk_t(
    char *uplo, char *trans, int *n, int *k,
    double *alpha, double *A, int *lda,
    double *beta, double *c, int *ldc
    ) nogil
cdef dsyrk_t *dsyrk

##################
#  example call  #
##################

# NOTE: passing typed memoryviews means passing structs, which get copied if functions aren't inlined
# AFAICT there's valid syntax for passing references instead but an error is thrown
# a tiny benchmark suggests the overhead is real

# TODO single precision versions

# TODO maybe I should just work with column-major matrices in cython... this is confusing

cdef inline void r_gemv(double[:,::1] A, double[::1] x, double[::1] out) nogil:
    cdef double alpha = 1., beta = 0.
    cdef int inc = 1
    dgemv("T", <int*> &A.shape[1], <int*> &A.shape[0],
            &alpha, &A[0,0], <int*> &A.shape[1],
            &x[0], &inc, &beta,
            &out[0], &inc)

cdef inline void r_gemm(double[:,::1] A, double[:,::1] B, double[:,::1] out) nogil:
    cdef double alpha = 1., beta = 0.
    cdef int inc = 1
    dgemm("N", "N", <int*> &B.shape[1], <int*> &A.shape[0], <int*> &A.shape[1],
            &alpha,
            &B[0,0], <int*> &B.shape[1],
            &A[0,0], <int*> &A.shape[1],
            &beta,
            &out[0,0], <int*> &out.shape[1])

cdef inline void r_symv(char *uplo, double[:,::1] A, double[::1] x, double[::1] out) nogil:
    cdef double alpha = 1., beta = 0.
    cdef int inc = 1
    dsymv(uplo, <int*> &A.shape[0],
            &alpha,
            &A[0,0], <int*> &A.shape[1],
            &x[0], &inc,
            &beta,
            &out[0], &inc)

cdef inline void r_symm(char *side, char *uplo, double[:,::1] A, double[:,::1] B, double[:,::1] out) nogil:
    cdef double alpha = 1., beta = 0.
    cdef int inc = 1
    cdef char myside = 'L' if side[0] == 'R' else 'R'
    dsymm(&myside, uplo, <int*> &out.shape[1], <int*> &out.shape[0],
            &alpha,
            &A[0,0], <int*> &A.shape[1],
            &B[0,0], <int*> &B.shape[1],
            &beta,
            &out[0,0], <int*> &out.shape[1])


#####################
#  extra functions  #
#####################

# computes out = A * X * A**T + alpha*out for row-major A and symmetric X and out
# TODO add uplo argument; currently requires X to be really symmetric; reads/writes both sides!
# TODO add trans argument (i.e. support col-major?)
cdef inline void matrix_qform(
        int m, int k,
        floating *A, # m x k
        floating *X, # k x k
        floating alpha, floating *out # m x m
        ):
    cdef int i, j, ii, jj
    cdef floating acc, ai, aj

    for i in range(m):
        for j in range(i,m):
            acc = 0.
            for ii in range(k):
                ai = A[j*k+ii]
                for jj in range(k):
                    aj = A[i*k+jj]
                    acc += ai * aj * X[ii*k+jj]
            out[i*m+j] = out[j*m+i] = acc + alpha*out[i*m+j]

# TODO for higher-rank updates, Householder reflections may be preferrable
cdef inline void chol_update(int n, floating *R, floating *z) nogil:
    cdef int k
    cdef int inc = 1
    cdef floating a, b, c, s
    if floating is double:
        for k in range(n):
            a, b = R[k*n+k], z[k]
            drotg(&a,&b,&c,&s)
            drot(&n,&R[k*n],&inc,&z[0],&inc,&c,&s)
    else:
        for k in range(n):
            a, b = R[k*n+k], z[k]
            srotg(&a,&b,&c,&s)
            srot(&n,&R[k*n],&inc,&z[0],&inc,&c,&s)

cdef inline void chol_downdate(int n, floating *R, floating *z) nogil:
    cdef int k, j
    cdef floating rbar
    for k in range(n):
        rbar = sqrt((R[k*n+k] - z[k])*(R[k*n+k] + z[k]))
        for j in range(k+1,n):
            R[k*n+j] = (R[k*n+k]*R[k*n+j] - z[k]*z[j]) / rbar
            z[j] = (rbar*z[j] - z[k]*R[k*n+j]) / R[k*n+k]
        R[k*n+k] = rbar


