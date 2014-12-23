from __future__ import division
import numpy as np
from numpy.random import randn
from scipy.linalg.blas import drot, drotg

# references for updates:
#   - Golub and van Loan (4th ed.) Section 6.5.4
#   - http://mathoverflow.net/questions/30162/is-there-a-way-to-simplify-block-cholesky-decomposition-if-you-already-have-deco
#
# references for downdates:
#   - Golub and van Loan (4th ed.) Section 6.5.4
#   - Alexander, Pan, and Plemmons, 1988
#     http://ac.els-cdn.com/0024379588901589/1-s2.0-0024379588901589-main.pdf?_tid=c35b3e32-8640-11e4-a30d-00000aacb35d&acdnat=1418857524_90c69b8bbe27d77d950c45d2c5d43410
# algorithm numbers refer to the APP98 paper

# other implementations:
#   - M. Seeger implementation is better than LINPACK mainly because it uses
#     BLAS, but it's all matlabby and also GPL:
#     http://www.ams.org/journals/mcom/1974-28-126/S0025-5718-1974-0343558-6/home.html
#     http://ipg.epfl.ch/~seeger/lapmalmainweb/software/index.shtml
#   - M. Hoffman wrapped the dchud/dchdd routines of LINPACK
#     https://github.com/mwhoffman/pychud

def update(R,z):
    n = z.shape[0]
    for k in range(n):
        c, s = drotg(R[k,k],z[k])
        drot(R[k,:],z,c,s,overwrite_x=True,overwrite_y=True)
    return R

# should be same as linpack's dchdd, uses orthogonal transformations
# should also be essentially the same as M. Seeger's GPL implementation
# requires a triangular solve and so it's supposedly not so good for
# parallel/vector machines
def algorithm1(R,z):
    n = R.shape[0]

    a = np.linalg.solve(R.T,z) # TODO could be triangular solve
    A = np.vstack((R,np.zeros((1,n))))
    for k in range(n-1,-1,-1):
        hk = np.sqrt(1 - a[:k].dot(a[:k]))
        ck = np.sqrt(1 - a[:k+1].dot(a[:k+1])) / hk
        sk = a[k] / hk

        A[k,:], A[-1,:] = ck*A[k,:] - sk*A[-1,:], sk*A[k,:] + ck*A[-1,:]

    return A[:-1]


# hyperbolic rotations
# NOTE: in-place!
def algorithm2(R,z):
    n = z.shape[0]

    for k in range(n):
        tk = z[k] / R[k,k]
        ck = 1./np.sqrt(1-tk**2)
        sk = ck*tk

        R[k,:], z[:] = ck*R[k,:] - sk*z, -sk*R[k,:] + ck*z

    return R

# hyperbolic rotations with a different and more stable computation
# NOTE: in-place!
def algorithm2prime(R,z):
    n = R.shape[0]

    for k in range(n):
        rbar = np.sqrt((R[k,k] - z[k])*(R[k,k] + z[k]))
        for j in range(k+1,n):
            R[k,j] = 1./rbar * (R[k,k]*R[k,j] - z[k]*z[j])
            z[j] = 1./R[k,k] * (rbar*z[j] - z[k]*R[k,j])
        R[k,k] = rbar

    return R

# downdate = algorithm1
downdate = algorithm2
# downdate = algorithm2prime

def test_downdate():

    A = randn(3,3)
    A = A.dot(A.T)
    A += 10*np.eye(3)

    v = randn(3)

    L = np.linalg.cholesky(A)
    Ltilde = np.linalg.cholesky(A - np.outer(v,v))

    Ltilde2 = algorithm1(L.T.copy(),v.copy()).T
    assert np.allclose(Ltilde,Ltilde2)

    Ltilde3 = algorithm2(L.T.copy(),v.copy()).T
    assert np.allclose(Ltilde,Ltilde3)

    Ltilde4 = algorithm2prime(L.T.copy(),v.copy()).T
    assert np.allclose(Ltilde,Ltilde4)

    from lds import downdate as downdate2
    Ltilde5 = downdate2(L.T.copy(),v.copy()).T
    assert np.allclose(Ltilde,Ltilde5)


def test_update():
    A = randn(3,3)
    A = A.dot(A.T)

    v = randn(3)

    L = np.linalg.cholesky(A)
    Ltilde = np.linalg.cholesky(A + np.outer(v,v))

    Ltilde2 = update(L.T.copy(),v.copy()).T
    assert np.allclose(Ltilde,Ltilde2)

    from lds import update as update2
    Ltilde3 = update(L.T.copy(),v.copy()).T

    import sys
    sys.stdout.flush()
    assert np.allclose(Ltilde,Ltilde3)

if __name__ == '__main__':
    test_downdate()
    test_update()
    print 'All tests passed'

