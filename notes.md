scipy access to blas/lapack:
http://docs.scipy.org/doc/scipy-dev/reference/linalg.blas.html
http://docs.scipy.org/doc/scipy-dev/reference/linalg.lapack.html


way to call from cython:
https://stackoverflow.com/questions/16114100/calling-dot-products-and-linear-algebra-operations-in-cython
https://gist.github.com/pv/5437087
https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/kalmanf/kalman_loglike.pyx#L26
https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/kalmanf/blas_lapack.pxd


i am not liking Eigen because i can't figure out how to perform the exact
solves that i want. furthermore, since scipy provides the best blas the system
has, it seems easier than trying to distribute lapack or link against a blas.
so cython wins here!



hmm, but with this strategy of pulling out the pointer, the compiler may not be
able to inline?
