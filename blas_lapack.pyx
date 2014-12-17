# distutils: extra_compile_args = -O3 -w
# cython: boundscheck = False
# cython: nonecheck = False
# cython: wraparound = False
# cython: cdivision = True

import scipy.linalg.blas
import scipy.linalg.lapack

cdef extern from "f2pyptr.h":
    void *f2py_pointer(object) except NULL

cdef:
    ### BLAS
    # level 2
    ddot_t *ddot = <ddot_t*>f2py_pointer(scipy.linalg.blas.ddot._cpointer)
    dger_t *dger = <dger_t*>f2py_pointer(scipy.linalg.blas.dger._cpointer)

    # level 3
    dgemv_t *dgemv = <dgemv_t*>f2py_pointer(scipy.linalg.blas.dgemv._cpointer)
    dgemm_t *dgemm = <dgemm_t*>f2py_pointer(scipy.linalg.blas.dgemm._cpointer)
    dsymv_t *dsymv = <dsymv_t*>f2py_pointer(scipy.linalg.blas.dsymv._cpointer)
    dsymm_t *dsymm = <dsymm_t*>f2py_pointer(scipy.linalg.blas.dsymm._cpointer)

    ### LAPACK
    dposv_t* dposv = <dposv_t*>f2py_pointer(scipy.linalg.lapack.dposv._cpointer)
    dpotrf_t* dpotrf = <dpotrf_t*>f2py_pointer(scipy.linalg.lapack.dpotrf._cpointer)
    dpotrs_t* dpotrs = <dpotrs_t*>f2py_pointer(scipy.linalg.lapack.dpotrs._cpointer)

