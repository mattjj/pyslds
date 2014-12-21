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
able to inline? whatever





cholesky up/downdating

there are linpack routines

seeger implementation is better than LINPACK mainly because it uses BLAS, but
it's all matlabby and also GPL:
http://www.ams.org/journals/mcom/1974-28-126/S0025-5718-1974-0343558-6/home.html
http://ipg.epfl.ch/~seeger/lapmalmainweb/software/index.shtml

discussion:
http://mathoverflow.net/questions/30162/is-there-a-way-to-simplify-block-cholesky-decomposition-if-you-already-have-deco

GvL 4th ed. Section 6.5 has a great discussion
For numerical stability, its reference Algorithm 2' is best:
http://www.sciencedirect.com/science/article/pii/0024379588901589
