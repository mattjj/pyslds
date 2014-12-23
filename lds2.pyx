# distutils: extra_compile_args = -O3 -w -DNDEBUG -std=c++11 -I/Users/mattjj/code/pyhsmm/internals/ -I/Users/mattjj/code/pyhsmm/deps/Eigen3/

cdef extern from "slds_util.h":
    cdef cppclass dummy[Type]:
        dummy()
        void condition_on(
            int D, int P,
            Type *mu_x, Type *sigma_x, Type *A, Type *sigma_obs, Type *y,
            Type *mu_out, Type *sigma_out) nogil

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

