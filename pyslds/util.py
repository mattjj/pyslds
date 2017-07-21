import numpy as np

from pybasicbayes.distributions.regression import AutoRegression

def get_empirical_ar_params(train_datas, params):
    """
    Estimate the parameters of an AR observation model
    by fitting a single AR model to the entire dataset.
    """
    assert isinstance(train_datas, list) and len(train_datas) > 0
    datadimension = train_datas[0].shape[1]
    assert params["nu_0"] > datadimension + 1

    # Initialize the observation parameters
    obs_params = dict(nu_0=params["nu_0"],
                      S_0=params['S_0'],
                      M_0=params['M_0'],
                      K_0=params['K_0'],
                      affine=params['affine'])


    # Fit an AR model to the entire dataset
    obs_distn = AutoRegression(**obs_params)
    obs_distn.max_likelihood(train_datas)

    # Use the inferred noise covariance as the prior mean
    # E_{IW}[S] = S_0 / (nu_0 - datadimension - 1)
    obs_params["S_0"] = obs_distn.sigma * (params["nu_0"] - datadimension - 1)
    obs_params["M_0"] = obs_distn.A.copy()

    return obs_params


def expected_hmm_logprob(pi_0, trans_matrix, stats):
    """
    :param pi_0:          initial distribution
    :param trans_matrix:  transition matrix
    :param stats:         tuple (E[z_t], \sum_t E[z_t z_{t+1}.T])

    :return:  E_{q(z)} [ log p(z) ]
    """
    E_z, sum_E_ztztp1T, _ = stats
    T, K = E_z.shape
    assert sum_E_ztztp1T.shape == (K, K)

    out = 0
    out += np.dot(E_z[0], np.log(pi_0))
    out += np.sum(sum_E_ztztp1T * np.log(trans_matrix))
    return out


def hmm_entropy(params, stats):
    log_transmatrix, log_pi_0, aBl, _ = params
    E_z, sum_E_ztztp1T, log_Z = stats
    T, K = E_z.shape
    assert aBl.shape == (T, K)
    assert sum_E_ztztp1T.shape == (K, K)
    assert log_transmatrix.shape == (K, K)

    neg_entropy = np.sum(E_z[0] * log_pi_0)
    neg_entropy += np.sum(E_z * aBl)
    neg_entropy += np.sum(sum_E_ztztp1T * log_transmatrix)
    neg_entropy -= log_Z
    return -neg_entropy


def expected_gaussian_logprob(mu, sigma, stats):
    D = mu.shape[0]

    J = np.linalg.inv(sigma)
    h = J.dot(mu)
    muJmuT = mu.dot(J).dot(mu.T)
    logdetJ = np.linalg.slogdet(J)[1]

    x, xxT, n = stats
    c1, c2 = ('i,i->', 'ij,ij->') if x.ndim == 1 \
        else ('i,ni->n', 'ij,nij->n')

    out = -1. / 2 * np.einsum(c2, J, xxT)
    out += np.einsum(c1, h, x)
    out += -n / 2. * muJmuT
    out += -D / 2. * np.log(2 * np.pi) + n / 2. * logdetJ

    return out


def expected_regression_log_prob(A, Sigma, stats):
    """
    Expected log likelihood of p(y | x) where

        y ~ N(Ax, Sigma)

    and expectation is wrt q(y,x).  We only need expected
    sufficient statistics E[yy.T], E[yx.T], E[xx.T], and n,
    where n is the number of observations.

    :param A:      regression matrix
    :param Sigma:  observation covariance
    :param stats:  tuple (E[yy.T], E[yx.T], E[xx.T], n)
    :return:       E[log p(y | x)]
    """
    yyT, yxT, xxT, n = stats[-4:]

    contract = 'ij,nij->n' if yyT.ndim == 3 else 'ij,ij->'
    D = A.shape[0]
    Si = np.linalg.inv(Sigma)
    SiA = Si.dot(A)
    ASiA = A.T.dot(SiA)

    out = -1. / 2 * np.einsum(contract, ASiA, xxT)
    out += np.einsum(contract, SiA, yxT)
    out += -1. / 2 * np.einsum(contract, Si, yyT)
    out += -D / 2 * np.log(2 * np.pi) + n / 2. * np.linalg.slogdet(Si)[1]
    return out


def lds_entropy(info_params, stats):
    # Extract the info params that make up the variational factor
    J_init, h_init, log_Z_init, \
    J_pair_11, J_pair_21, J_pair_22, h_pair_1, h_pair_2, log_Z_pair, \
    J_node, h_node, log_Z_node = info_params

    # Extract the expected sufficient statistics
    _lds_normalizer, E_x, Var_x, E_xtp1_xtT = stats
    E_x_xT = Var_x + E_x[:, :, None] * E_x[:, None, :]

    contract = 'tij,tji->'

    # Initial potential
    nep = -1. / 2 * np.sum(J_init * E_x_xT[0])
    nep += h_init.dot(E_x[0])
    nep += log_Z_init

    # Pair potentials
    nep += -1. / 2 * np.einsum(contract, J_pair_22, E_x_xT[1:])
    nep += - np.einsum(contract, np.swapaxes(J_pair_21, 1, 2), E_xtp1_xtT)
    nep += -1. / 2 * np.einsum(contract, J_pair_11, E_x_xT[:-1])
    nep += np.sum(h_pair_1 * E_x[:-1])
    nep += np.sum(h_pair_2 * E_x[1:])
    nep += np.sum(log_Z_pair)

    # Node potentials
    nep += -1. / 2 * np.einsum(contract, J_node, E_x_xT)
    nep += np.sum(h_node * E_x)
    nep += np.sum(log_Z_node)

    # Normalizer
    nep += -_lds_normalizer
    return -nep


def symmetric_blk_tridiagonal_logdet(diagonal_array, off_diagonal_array):
    T = len(diagonal_array)
    n = diagonal_array.shape[1]

    J = np.zeros((T * n, T * n))
    for t in np.arange(T):
        J[t * n: t * n + n, t * n: t * n + n] = diagonal_array[t]
    for t in np.arange(T-1):
        J[t * n: t * n + n, t * n + n: t * n + 2 * n] += off_diagonal_array[t].T
        J[t * n + n: t * n + 2 * n, t * n: t * n + n] += off_diagonal_array[t]
    return np.linalg.slogdet(J)[1]


def test_lds_entropy(info_params):
    # Extract the info params that make up the variational factor
    J_init, h_init, log_Z_init, \
    J_pair_11, J_pair_21, J_pair_22, h_pair_1, h_pair_2, log_Z_pair, \
    J_node, h_node, log_Z_node = info_params

    T, D = h_node.shape

    # Compute the variational entropy by constructing the full Gaussian params.
    diagonal_array = J_node.copy()
    diagonal_array[0] += J_init
    diagonal_array[:-1] += J_pair_11
    diagonal_array[1:] += J_pair_22
    off_diagonal_array = J_pair_21.copy()
    ve = -1. / 2 * symmetric_blk_tridiagonal_logdet(diagonal_array, off_diagonal_array)
    ve += 1. / 2 * T * D * (1 + np.log(2 * np.pi))
    return ve


def gaussian_map_estimation(stats, gaussian):
    D = gaussian.D
    x, xxT, n = stats

    # Add "pseudocounts" from the prior
    mu_0, sigma_0, kappa_0, nu_0 = \
        gaussian.mu_0, gaussian.sigma_0, gaussian.kappa_0, gaussian.nu_0

    xxT += sigma_0 + kappa_0 * np.outer(mu_0, mu_0)
    x += kappa_0 * mu_0
    n += nu_0 + 2 + D

    # SVD is necessary to check if the max likelihood solution is
    # degenerate, which can happen in the EM algorithm
    if n < D or (np.linalg.svd(xxT, compute_uv=False) > 1e-6).sum() < D:
        raise Exception("Can't to MAP when effective observations < D")

    # Set the MAP params
    gaussian.mu = x / n
    gaussian.sigma = xxT / n - np.outer(gaussian.mu, gaussian.mu)


def regression_map_estimation(stats, regression):
    D_out = regression.D_out

    # Add prior and likelihood statistics
    sum_tuples = lambda lst: list(map(sum, zip(*lst)))
    yyT, yxT, xxT, n = sum_tuples([stats, regression.natural_hypparam])

    A = np.linalg.solve(xxT, yxT.T).T
    sigma = (yyT - A.dot(yxT.T)) / n

    # Make sure sigma is symmetric
    symmetrize = lambda A: (A + A.T) / 2.
    sigma = 1e-10 * np.eye(D_out) + symmetrize(sigma)

    regression.A = A
    regression.sigma = sigma


def niw_logprob(gaussian):
    D = gaussian.D
    mu, sigma = gaussian.mu, gaussian.sigma
    mu_0, sigma_0, kappa_0, nu_0 = \
        gaussian.mu_0, gaussian.sigma_0, gaussian.kappa_0, gaussian.nu_0

    # Inverse Wishart  IW(sigma | sigma_0, nu_0)
    from pybasicbayes.util.stats import invwishart_log_partitionfunction
    lp = invwishart_log_partitionfunction(sigma_0, nu_0)
    lp += -(nu_0 + D + 1) / 2.0 * np.linalg.slogdet(sigma)[1]
    lp += -0.5 * np.trace(np.linalg.solve(sigma, sigma_0))

    # Normal N(mu | mu_0, Sigma / kappa_0)
    from scipy.linalg import solve_triangular
    S_chol = np.linalg.cholesky(sigma / kappa_0)
    x = solve_triangular(S_chol, mu - mu_0, lower=True)
    lp += -1. / 2. * np.dot(x, x) \
          - D / 2 * np.log(2 * np.pi) \
          - np.log(S_chol.diagonal()).sum()

    return lp


def mniw_logprob(regression):
    A = regression.A
    Sigmainv = np.linalg.inv(regression.sigma)
    Sigmainv_A = Sigmainv.dot(A)
    AT_Sigmainv_A = A.T.dot(Sigmainv_A)
    logdetSigmainv = np.linalg.slogdet(Sigmainv)[1]

    A, B, C, d = regression.natural_hypparam
    bilinear_term = -1./2 * np.trace(A.dot(Sigmainv)) \
        + np.trace(B.T.dot(Sigmainv_A)) \
        - 1./2 * np.trace(C.dot(AT_Sigmainv_A)) \
        + 1./2 * d * logdetSigmainv

    # log normalizer term
    from pybasicbayes.util.stats import mniw_log_partitionfunction
    Z = mniw_log_partitionfunction(
        *regression._natural_to_standard(regression.natural_hypparam))

    return bilinear_term - Z
