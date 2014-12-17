### BLAS http://www.netlib.org/blas/

# http://www.netlib.org/lapack/explore-html/d5/df6/ddot_8f.html
ctypedef double ddot_t(
    int *n, double *x, int *incx, double *y, int *incy
    ) nogil

# http://www.netlib.org/lapack/explore-html/dc/da8/dgemv_8f.html
ctypedef int dgemv_t(
    char *trans,
    int *m, int *n, double *alpha, double *A, int *lda,
    double *x, int *incx,
    double *beta, double *y, int *incy
    ) nogil

# http://www.netlib.org/lapack/explore-html/d7/d2b/dgemm_8f.html
ctypedef int dgemm_t(
    char *transa, char *transb, int *m, int *n, int *k,
    double *alpha, double *A, int *lda,
    double *B, int *ldb,
    double *beta, double *C, int *ldc
    ) nogil

# http://www.netlib.org/lapack/explore-html/dc/da8/dsymv_8f.html
ctypedef int dsymv_t(
    char *uplo, int *n,
    double *alpha, double *A, int *lda,
    double *x, int *incx,
    double *beta, double *y, int *incy
    ) nogil

# http://www.netlib.org/lapack/explore-html/d8/db0/dsymm_8f.html
ctypedef int dsymm_t(
    char *side, char *uplo, int *m, int *n,
    double *alpha, double *A, int *lda,
    double *B, int *ldb,
    double *beta, double *C, int *ldc
    ) nogil

# http://www.netlib.org/lapack/explore-html/dc/da8/dger_8f.html
ctypedef double dger_t(
    int *m, int *n, double *alpha,
    double *x, int *incx,
    double *y, int *incy,
    double *A, int *lda
    ) nogil

### LAPACK http://www.netlib.org/lapack/

ctypedef int dposv_t(
    char *uplo, int *n, int *nrhs,
    double *A, int *lda,
    double *B, int *ldb,
    int *info) nogil

ctypedef int dpotrf_t(
    char *uplo, int *n,
    double *a, int *lda,
    int *info) nogil

ctypedef int dpotrs_t(
    char *uplo, int *n, int *nrhs,
    double *a, int *lda,
    double *b, int *ldb,
    int *info) nogil

cdef inline void dotmv(double[:,::1] A, double[::1] x, double[::1] out)

