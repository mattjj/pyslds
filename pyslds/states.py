from __future__ import division
import numpy as np
from functools import partial

from pyhsmm.models import HMM

from pyhsmm.internals.hmm_states import HMMStatesPython, HMMStatesEigen
from pyhsmm.internals.hsmm_states import HSMMStatesPython, HSMMStatesEigen, \
    GeoHSMMStates

from pylds.lds_messages_interface import filter_and_sample, info_E_step, info_sample

from pyslds.util import hmm_entropy, lds_entropy, expected_regression_log_prob, expected_gaussian_logprob, test_lds_entropy

import pypolyagamma as ppg
from pypolyagamma.distributions import _PGLogisticRegressionBase

# TODO on instantiating, maybe gaussian states should be resampled
# TODO make niter an __init__ arg instead of a method arg


###########
#  bases  #
###########

class _SLDSStates(object):
    def __init__(self,model,T=None,data=None,inputs=None,stateseq=None,gaussian_states=None,
            generate=True,initialize_from_prior=True,fixed_stateseq=None):
        self.model = model

        self.T = T if T is not None else data.shape[0]
        self.data = data
        self.inputs = np.zeros((self.T, 0)) if inputs is None else inputs
        self.fixed_stateseq = fixed_stateseq
        self.clear_caches()

        # store gaussian states and state sequence if passed in
        if gaussian_states is not None and stateseq is not None:
            self.gaussian_states = gaussian_states
            self.stateseq = np.array(stateseq,dtype=np.int32)
            if data is not None and not initialize_from_prior:
                self.resample()

        elif stateseq is not None:
            self.stateseq = np.array(stateseq,dtype=np.int32)
            self.generate_gaussian_states()

        elif generate:
            if data is not None and not initialize_from_prior:
                self.resample()
            else:
                self.generate_states()

    def generate_states(self):
        super(_SLDSStates,self).generate_states()
        self.generate_gaussian_states()

    def generate_gaussian_states(self):
        # Generate from the prior and raise exception if unstable
        T, n = self.T, self.D_latent

        # The discrete stateseq should be populated by the super call above
        dss = self.stateseq

        gss = np.empty((T,n),dtype='double')
        gss[0] = self.init_dynamics_distns[dss[0]].rvs()

        for t in range(1,T):
            gss[t] = self.dynamics_distns[dss[t]].\
                rvs(x=np.hstack((gss[t-1][None,:], self.inputs[t-1][None,:])),
                    return_xy=False)
            assert np.all(np.isfinite(gss[t])), "SLDS appears to be unstable!"

        self.gaussian_states = gss

    def generate_obs(self):
        # Go through each time bin, get the discrete latent state,
        # use that to index into the emission_distns to get samples
        T, p = self.T, self.D_emission
        dss, gss = self.stateseq, self.gaussian_states
        data = np.empty((T,p),dtype='double')

        for t in range(self.T):
            ed = self.emission_distns[0] if self.model._single_emission \
                else self.emission_distns[dss[t]]
            data[t] = \
                ed.rvs(x=np.hstack((gss[t][None, :], self.inputs[t][None,:])),
                       return_xy=False)

        return data

    ## convenience properties

    @property
    def D_latent(self):
        return self.dynamics_distns[0].D_out

    @property
    def D_input(self):
        return self.dynamics_distns[0].D_in - self.dynamics_distns[0].D_out

    @property
    def D_emission(self):
        return self.emission_distns[0].D_out

    @property
    def dynamics_distns(self):
        return self.model.dynamics_distns

    @property
    def emission_distns(self):
        return self.model.emission_distns

    @property
    def init_dynamics_distns(self):
        return self.model.init_dynamics_distns

    @property
    def diagonal_noise(self):
        return self.model.diagonal_noise

    @property
    def mu_init(self):
        return self.init_dynamics_distns[self.stateseq[0]].mu

    @property
    def sigma_init(self):
        return self.init_dynamics_distns[self.stateseq[0]].sigma

    @property
    def As(self):
        Aset = np.concatenate([d.A[None,:,:self.D_latent] for d in self.dynamics_distns])
        return Aset[self.stateseq]

    @property
    def Bs(self):
        Bset = np.concatenate([d.A[None,:,self.D_latent:] for d in self.dynamics_distns])
        return Bset[self.stateseq]

    @property
    def sigma_statess(self):
        sset = np.concatenate([d.sigma[None,...] for d in self.dynamics_distns])
        return sset[self.stateseq]

    @property
    def Cs(self):
        Cset = np.concatenate([d.A[None,:,:self.D_latent] for d in self.emission_distns])
        return Cset[self.stateseq]

    @property
    def Ds(self):
        Dset = np.concatenate([d.A[None, :, self.D_latent:] for d in self.emission_distns])
        return Dset[self.stateseq]

    @property
    def sigma_obss(self):
        sset = np.concatenate([d.sigma[None,...] for d in self.emission_distns])
        return sset[self.stateseq]

    @property
    def _kwargs(self):
        return dict(super(_SLDSStates, self)._kwargs,
                    stateseq=self.stateseq,
                    gaussian_states=self.gaussian_states)

    @property
    def info_init_params(self):
        J_init = np.linalg.inv(self.sigma_init)
        h_init = np.linalg.solve(self.sigma_init, self.mu_init)

        log_Z_init = -1. / 2 * h_init.dot(np.linalg.solve(J_init, h_init))
        log_Z_init += 1. / 2 * np.linalg.slogdet(J_init)[1]
        log_Z_init -= self.D_latent / 2. * np.log(2 * np.pi)

        return J_init, h_init, log_Z_init

    @property
    def info_dynamics_params(self):
        z, u = self.stateseq[:-1], self.inputs[:-1]
        expand = lambda a: a[None,...]
        stack_set = lambda x: np.concatenate(list(map(expand, x)))

        A_set = [d.A[:,:self.D_latent] for d in self.dynamics_distns]
        B_set = [d.A[:,self.D_latent:] for d in self.dynamics_distns]
        Q_set = [d.sigma for d in self.dynamics_distns]

        # Get the pairwise potentials
        # TODO: Check for diagonal before inverting
        J_pair_22_set = [np.linalg.inv(Q) for Q in Q_set]
        J_pair_21_set = [-J22.dot(A) for A,J22 in zip(A_set, J_pair_22_set)]
        J_pair_11_set = [A.T.dot(-J21) for A,J21 in zip(A_set, J_pair_21_set)]

        J_pair_11 = stack_set(J_pair_11_set)[z]
        J_pair_21 = stack_set(J_pair_21_set)[z]
        J_pair_22 = stack_set(J_pair_22_set)[z]

        # Check if diagonal and avoid inverting D_obs x D_obs matrix
        h_pair_1_set = [B.T.dot(J) for B, J in zip(B_set, J_pair_21_set)]
        h_pair_2_set = [B.T.dot(Qi) for B, Qi in zip(B_set, J_pair_22_set)]

        h_pair_1 = stack_set(h_pair_1_set)[z]
        h_pair_2 = stack_set(h_pair_2_set)[z]

        h_pair_1 = np.einsum('ni,nij->nj', u, h_pair_1)
        h_pair_2 = np.einsum('ni,nij->nj', u, h_pair_2)

        # Compute the log normalizer
        log_Z_pair = -self.D_latent / 2. * np.log(2 * np.pi) * np.ones(self.T-1)

        logdet = [np.linalg.slogdet(Q)[1] for Q in Q_set]
        logdet = stack_set(logdet)[z]
        log_Z_pair += -1. / 2 * logdet

        hJh_pair = [B.T.dot(np.linalg.solve(Q, B)) for B, Q in zip(B_set, Q_set)]
        hJh_pair = stack_set(hJh_pair)[z]
        log_Z_pair -= 1. / 2 * np.einsum('tij,ti,tj->t', hJh_pair, u, u)

        return J_pair_11, J_pair_21, J_pair_22, h_pair_1, h_pair_2, log_Z_pair

    @property
    def info_emission_params(self):

        expand = lambda a: a[None,...]
        stack_set = lambda x: np.concatenate(list(map(expand, x)))

        # TODO: Check for diagonal emissions
        C_set = [d.A[:,:self.D_latent] for d in self.emission_distns]
        D_set = [d.A[:,self.D_latent:] for d in self.emission_distns]
        R_set = [d.sigma for d in self.emission_distns]
        Ri_set = [np.linalg.inv(R) for R in R_set]
        RiC_set = [Ri.dot(C) for C,Ri in zip(C_set, Ri_set)]
        RiD_set = [Ri.dot(D) for D,Ri in zip(D_set, Ri_set)]
        CRiC_set = [C.T.dot(RiC) for C,RiC in zip(C_set, RiC_set)]
        DRiC_set = [D.T.dot(RiC) for D,RiC in zip(D_set, RiC_set)]
        DRiD_set = [D.T.dot(RiD) for D,RiD in zip(D_set, RiD_set)]

        # TODO: Faster to replace this with a loop over t?
        Ri = stack_set(Ri_set)[self.stateseq]
        RiC = stack_set(RiC_set)[self.stateseq]
        RiD = stack_set(RiD_set)[self.stateseq]
        DRiC = stack_set(DRiC_set)[self.stateseq]
        DRiD = stack_set(DRiD_set)[self.stateseq]

        J_node = stack_set(CRiC_set)[self.stateseq]

        # h_node = y^T R^{-1} C - u^T D^T R^{-1} C
        h_node = np.einsum('ni,nij->nj', self.data, RiC)
        h_node -= np.einsum('ni,nij->nj', self.inputs, DRiC)

        log_Z_node = -self.D_emission / 2. * np.log(2 * np.pi) * np.ones(self.T)

        logdet = [np.linalg.slogdet(R)[1] for R in R_set]
        logdet = stack_set(logdet)[self.stateseq]
        log_Z_node += -1. / 2 * logdet

        # E[(y-Du)^T R^{-1} (y-Du)]
        log_Z_node -= 1. / 2 * np.einsum('tij,ti,tj->t', Ri,
                                         self.data, self.data)
        log_Z_node -= 1. / 2 * np.einsum('tij,ti,tj->t', -2 * RiD,
                                         self.data, self.inputs)
        log_Z_node -= 1. / 2 * np.einsum('tij,ti,tj->t', DRiD,
                                         self.inputs, self.inputs)

        # Observations
        # if self.diagonal_noise:
        #     # Use the fact that the diagonalregression prior is factorized
        #     rsq_set = [d.sigmasq_flat for d in self.emission_distns]
        #     rsq = stack_set(rsq_set)[self.stateseq]
        #
        #     J_yy = 1./rsq
        #     logdet_node = -np.sum(np.log(rsq), axis=1)
        #
        #     # We need terms for u_t D^T R^{-1} D u
        #     hJh_node_set = [D.T.dot(np.diag(1./r)).dot(D) for D, r in zip(D_set, rsq_set)]
        #     hJh_nodes = stack_set(hJh_node_set)[self.stateseq]


        return J_node, h_node, log_Z_node

    @property
    def info_params(self):
        return self.info_init_params + self.info_dynamics_params + self.info_emission_params

    # todo: reconsider smoothing interface.
    def smooth(self):
        # Use the info E step because it can take advantage of diagonal noise
        # The standard E step could but we have not implemented it
        self.info_E_step()
        xu = np.column_stack((self.smoothed_mus, self.inputs))
        if self.model._single_emission:
            return xu.dot(self.emission_distns[0].A.T)
        else:
            return np.array([C.dot(x) for C, x in zip(self.Cs, xu)])

    def info_E_step(self):
        self._gaussian_normalizer, self.smoothed_mus, \
        self.smoothed_sigmas, E_xtp1_xtT = \
            info_E_step(*self.info_params)

    def _init_mf_from_gibbs(self):
        # Base class sets the expected HMM stats
        # the first meanfield step will update the HMM params accordingly
        super(_SLDSStates, self)._init_mf_from_gibbs()

        self._normalizer = None
        self._mf_lds_normalizer = 0
        self.smoothed_mus = self.gaussian_states.copy()
        self.smoothed_sigmas = np.tile(0.01 * np.eye(self.D_latent)[None, :, :], (self.T, 1, 1))
        E_xtp1_xtT = self.smoothed_mus[1:,:,None] * self.smoothed_mus[:-1,None,:]

        self._set_gaussian_expected_stats(
            self.smoothed_mus, self.smoothed_sigmas, E_xtp1_xtT)

    def _set_gaussian_expected_stats(self, smoothed_mus, smoothed_sigmas, E_xtp1_xtT):
        """
        Both meanfield and VBEM require expected statistics of the continuous latent
        states, x.  This is a helper function to take E[x_t], E[x_t x_t^T] and E[x_{t+1}, x_t^T]
        and compute the expected sufficient statistics for the initial distribution,
        dynamics distribution, and Gaussian observation distribution.
        """
        assert not np.isnan(E_xtp1_xtT).any()
        assert not np.isnan(smoothed_mus).any()
        assert not np.isnan(smoothed_sigmas).any()
        assert smoothed_mus.shape == (self.T, self.D_latent)
        assert smoothed_sigmas.shape == (self.T, self.D_latent, self.D_latent)
        assert E_xtp1_xtT.shape == (self.T-1, self.D_latent, self.D_latent)

        # This is like LDSStates._set_expected_states but doesn't sum over time
        T = self.T
        E_x_xT = smoothed_sigmas + smoothed_mus[:, :, None] * smoothed_mus[:, None, :]
        E_x_uT = smoothed_mus[:, :, None] * self.inputs[:, None, :]
        E_u_uT = self.inputs[:, :, None] * self.inputs[:, None, :]

        E_xu_xuT = np.concatenate((
            np.concatenate((E_x_xT, E_x_uT), axis=2),
            np.concatenate((np.transpose(E_x_uT, (0, 2, 1)), E_u_uT), axis=2)),
            axis=1)
        E_xut_xutT = E_xu_xuT[:-1]
        E_xtp1_xtp1T = E_x_xT[1:]
        E_xtp1_utT = (smoothed_mus[1:, :, None] * self.inputs[:-1, None, :])
        E_xtp1_xutT = np.concatenate((E_xtp1_xtT, E_xtp1_utT), axis=-1)

        # Initial state stats
        self.E_init_stats = (self.smoothed_mus[0], E_x_xT[0], 1.)

        # Dynamics stats
        self.E_dynamics_stats = (E_xtp1_xtp1T, E_xtp1_xutT, E_xut_xutT, np.ones(self.T-1))

        # Emission stats
        E_yyT = self.data**2 if self.diagonal_noise else self.data[:, :, None] * self.data[:, None, :]
        E_yxT = self.data[:, :, None] * self.smoothed_mus[:, None, :]
        E_yuT = self.data[:, :, None] * self.inputs[:, None, :]
        E_yxuT = np.concatenate((E_yxT, E_yuT), axis=-1)
        self.E_emission_stats = (E_yyT, E_yxuT, E_xu_xuT, np.ones(T))

######################
#  algorithm mixins  #
######################

class _SLDSStatesGibbs(_SLDSStates):
    def resample(self, niter=1):
        niter = self.niter if hasattr(self, 'niter') else niter
        for itr in range(niter):
            self.resample_discrete_states()
            self.resample_gaussian_states()

    def _init_gibbs_from_mf(self):
        raise NotImplementedError  # TODO

    def resample_discrete_states(self):
        super(_SLDSStatesGibbs, self).resample()

    def resample_gaussian_states(self):
        self._aBl = None  # clear any caching
        self._gaussian_normalizer, self.gaussian_states = \
            info_sample(*self.info_params)


class _SLDSStatesVBEM(_SLDSStates):
    def __init__(self, model, **kwargs):
        super(_SLDSStatesVBEM, self).__init__(model, **kwargs)
        self.smoothed_mus = np.zeros((self.T, self.D_latent))
        self.smoothed_sigmas = np.tile(np.eye(self.D_latent)[None, :, :], (self.T, self.D_latent))

    @property
    def vbem_info_init_params(self):
        E_z0 = self.expected_states[0]
        expand = lambda a: a[None, ...]
        stack_set = lambda x: np.concatenate(list(map(expand, x)))

        mu_set = [d.mu for d in self.init_dynamics_distns]
        sigma_set = [d.sigma for d in self.init_dynamics_distns]

        J_init_set = stack_set([np.linalg.inv(sigma) for sigma in sigma_set])
        h_init_set = stack_set([J.dot(mu) for J, mu in zip(J_init_set, mu_set)])

        J_init = np.tensordot(E_z0, J_init_set, axes=1)
        h_init = np.tensordot(E_z0, h_init_set, axes=1)

        hJh_init = np.array([h.T.dot(S).dot(h) for S, h in zip(sigma_set, h_init_set)])
        logdet = np.array([np.linalg.slogdet(J)[1] for J in J_init_set])
        log_Z_init = -1. / 2 * np.dot(E_z0, hJh_init)
        log_Z_init += 1. / 2 * np.dot(E_z0, logdet)
        log_Z_init -= self.D_latent / 2. * np.log(2 * np.pi)

        return J_init, h_init, log_Z_init

    @property
    def vbem_info_dynamics_params(self):
        E_z = self.expected_states[:-1]
        expand = lambda a: a[None, ...]
        stack_set = lambda x: np.concatenate(list(map(expand, x)))

        A_set = [d.A[:, :self.D_latent] for d in self.dynamics_distns]
        B_set = [d.A[:, self.D_latent:] for d in self.dynamics_distns]
        Q_set = [d.sigma for d in self.dynamics_distns]

        # Get the pairwise potentials
        # TODO: Check for diagonal before inverting
        J_pair_22_set = stack_set([np.linalg.inv(Q) for Q in Q_set])
        J_pair_21_set = stack_set([-J22.dot(A) for A, J22 in zip(A_set, J_pair_22_set)])
        J_pair_11_set = stack_set([A.T.dot(-J21) for A, J21 in zip(A_set, J_pair_21_set)])

        J_pair_22 = np.tensordot(E_z, J_pair_22_set, axes=1)
        J_pair_21 = np.tensordot(E_z, J_pair_21_set, axes=1)
        J_pair_11 = np.tensordot(E_z, J_pair_11_set, axes=1)

        h_pair_1_set = stack_set([B.T.dot(J) for B, J in zip(B_set, J_pair_21_set)])
        h_pair_2_set = stack_set([B.T.dot(Qi) for B, Qi in zip(B_set, J_pair_22_set)])

        h_pair_1 = np.tensordot(E_z, h_pair_1_set, axes=1)
        h_pair_2 = np.tensordot(E_z, h_pair_2_set, axes=1)

        h_pair_1 = np.einsum('ni,nij->nj', self.inputs[:-1], h_pair_1)
        h_pair_2 = np.einsum('ni,nij->nj', self.inputs[:-1], h_pair_2)

        # Compute the log normalizer
        log_Z_pair = -self.D_latent / 2. * np.log(2 * np.pi) * np.ones(self.T - 1)

        logdet = np.array([np.linalg.slogdet(Q)[1] for Q in Q_set])
        logdet = np.dot(E_z, logdet)
        log_Z_pair += -1. / 2 * logdet

        hJh_pair = np.array([B.T.dot(np.linalg.solve(Q, B)) for B, Q in zip(B_set, Q_set)])
        hJh_pair = np.tensordot(E_z, hJh_pair, axes=1)
        log_Z_pair -= 1. / 2 * np.einsum('tij,ti,tj->t',
                                         hJh_pair,
                                         self.inputs[:-1],
                                         self.inputs[:-1])

        return J_pair_11, J_pair_21, J_pair_22, h_pair_1, h_pair_2, log_Z_pair

    @property
    def vbem_info_emission_params(self):

        expand = lambda a: a[None, ...]
        stack_set = lambda x: np.concatenate(list(map(expand, x)))

        # TODO: Check for diagonal emissions
        C_set = stack_set([d.A[:, :self.D_latent] for d in self.emission_distns])
        D_set = stack_set([d.A[:, self.D_latent:] for d in self.emission_distns])
        R_set = stack_set([d.sigma for d in self.emission_distns])
        Ri_set = stack_set([np.linalg.inv(R) for R in R_set])
        RiC_set = stack_set([Ri.dot(C) for C, Ri in zip(C_set, Ri_set)])
        RiD_set = stack_set([Ri.dot(D) for D, Ri in zip(D_set, Ri_set)])
        CRiC_set = stack_set([C.T.dot(RiC) for C, RiC in zip(C_set, RiC_set)])
        DRiC_set = stack_set([D.T.dot(RiC) for D, RiC in zip(D_set, RiC_set)])
        DRiD_set = stack_set([D.T.dot(RiD) for D, RiD in zip(D_set, RiD_set)])

        # TODO: Faster to replace this with a loop over t?
        E_z = self.expected_states
        Ri = np.tensordot(E_z, Ri_set, axes=1)
        RiC = np.tensordot(E_z, RiC_set, axes=1)
        RiD = np.tensordot(E_z, RiD_set, axes=1)
        DRiC = np.tensordot(E_z, DRiC_set, axes=1)
        DRiD = np.tensordot(E_z, DRiD_set, axes=1)

        J_node = np.tensordot(E_z, CRiC_set, axes=1)

        # h_node = y^T R^{-1} C - u^T D^T R^{-1} C
        h_node = np.einsum('ni,nij->nj', self.data, RiC)
        h_node -= np.einsum('ni,nij->nj', self.inputs, DRiC)

        log_Z_node = -self.D_emission / 2. * np.log(2 * np.pi) * np.ones(self.T)

        logdet = np.array([np.linalg.slogdet(R)[1] for R in R_set])
        logdet = np.dot(E_z, logdet)
        log_Z_node += -1. / 2 * logdet

        # E[(y-Du)^T R^{-1} (y-Du)]
        log_Z_node -= 1. / 2 * np.einsum('tij,ti,tj->t', Ri,
                                         self.data, self.data)
        log_Z_node -= 1. / 2 * np.einsum('tij,ti,tj->t', -2 * RiD,
                                         self.data, self.inputs)
        log_Z_node -= 1. / 2 * np.einsum('tij,ti,tj->t', DRiD,
                                         self.inputs, self.inputs)

        return J_node, h_node, log_Z_node

    @property
    def vbem_info_params(self):
        return self.vbem_info_init_params + \
               self.vbem_info_dynamics_params + \
               self.vbem_info_emission_params

    @property
    def vbem_aBl(self):
        """
        These are the expected log likelihoods (node potentials)
        as seen from the discrete states.  In other words,
        E_{q(x)} [log p(y, x | z)]
        """
        vbem_aBl = np.zeros((self.T, self.num_states))
        ids, dds, eds = self.init_dynamics_distns, self.dynamics_distns, \
            self.emission_distns

        for k, (id, dd, ed) in enumerate(zip(ids, dds, eds)):
            vbem_aBl[0, k] = expected_gaussian_logprob(id.mu, id.sigma, self.E_init_stats)
            vbem_aBl[:-1, k] += expected_regression_log_prob(dd.A, dd.sigma, self.E_dynamics_stats)
            vbem_aBl[:, k] += expected_regression_log_prob(ed.A, ed.sigma, self.E_emission_stats)

        vbem_aBl[np.isnan(vbem_aBl).any(1)] = 0.
        return vbem_aBl

    def vb_E_step(self):
        H_z = self.vb_E_step_discrete_states()
        H_x = self.vb_E_step_gaussian_states()
        self._variational_entropy = H_z + H_x

    def vb_E_step_discrete_states(self):
        # Call pyhsmm to do message passing and compute expected suff stats
        aBl = self.vbem_aBl
        self.all_expected_stats = self._expected_statistics(self.trans_matrix, self.pi_0, aBl)
        params = (np.log(self.trans_matrix), np.log(self.pi_0), aBl, self._normalizer)
        return hmm_entropy(params, self.all_expected_stats)

    def vb_E_step_gaussian_states(self):
        info_params = self.vbem_info_params

        # Call pylds to do message passing and compute expected suff stats
        stats = info_E_step(*info_params)
        self._lds_normalizer, self.smoothed_mus, self.smoothed_sigmas, E_xtp1_xtT = stats
        self._set_gaussian_expected_stats(self.smoothed_mus, self.smoothed_sigmas, E_xtp1_xtT)

        # Set the gaussian states to smoothed mus
        self.gaussian_states = self.smoothed_mus

        # Compute the variational entropy
        # ve1 = lds_entropy(info_params, stats)
        # ve2 = test_lds_entropy(info_params)
        # assert np.allclose(ve1, ve2)
        # return ve1
        return lds_entropy(info_params, stats)

    def vb_elbo(self):
        return self.expected_log_joint_probability() + self._variational_entropy

    def expected_log_joint_probability(self):
        """
        Compute E_{q(z) q(x)} [log p(z) + log p(x | z) + log p(y | x, z)]
        """
        # E_{q(z)}[log p(z)]
        from pyslds.util import expected_hmm_logprob
        elp = expected_hmm_logprob(
            self.pi_0, self.trans_matrix,
            (self.expected_states, self.expected_transcounts, self._normalizer))

        # E_{q(x)}[log p(y, x | z)]  is given by aBl
        # To get E_{q(x)}[ aBl ] we multiply and sum
        elp += np.sum(self.expected_states * self.vbem_aBl)
        return elp


class _SLDSStatesMeanField(_SLDSStates):
    def __init__(self, model, **kwargs):
        super(_SLDSStatesMeanField, self).__init__(model, **kwargs)
        self.smoothed_mus = np.zeros((self.T, self.D_latent))
        self.smoothed_sigmas = np.tile(np.eye(self.D_latent)[None, :, :], (self.T, self.D_latent))

    @property
    def expected_info_init_params(self):
        from pybasicbayes.util.stats import niw_expectedstats
        def get_paramseq(distns):
            contract = partial(np.tensordot, self.expected_states[0], axes=1)
            params = [niw_expectedstats(d.nu_mf, d.sigma_mf, d.mu_mf, d.kappa_mf)
                      for d in distns]
            return list(map(contract, zip(*params)))

        J_init, h_init, hJih_init, logdet_J_init = \
            get_paramseq(self.init_dynamics_distns)

        log_Z_init = -1. / 2 * hJih_init
        log_Z_init += 1. / 2 * logdet_J_init
        log_Z_init -= self.D_latent / 2. * np.log(2 * np.pi)

        return J_init, h_init, log_Z_init

    @property
    def expected_info_dynamics_params(self):
        def get_paramseq(distns):
            contract = partial(np.tensordot, self.expected_states[:-1], axes=1)
            params = [d.meanfield_expectedstats() for d in distns]
            return list(map(contract, zip(*params)))

        J_pair_22, J_pair_21, J_pair_11, logdet_pair = \
            get_paramseq(self.dynamics_distns)

        # Compute E[B^T Q^{-1}] and E[B^T Q^{-1} A]
        n = self.D_latent
        E_Qinv = J_pair_22
        E_Qinv_A = J_pair_21[:,:,:n]
        E_Qinv_B = J_pair_21[:,:,n:]
        E_BT_Qinv_A = J_pair_11[:,n:,:n]
        E_BT_Qinv_B = J_pair_11[:,n:,n:]
        E_AT_Qinv_A = J_pair_11[:,:n,:n].copy("C")

        h_pair_1 = -np.einsum('ti,tij->tj', self.inputs[:-1], E_BT_Qinv_A)
        h_pair_2 = np.einsum('ti,tji->tj', self.inputs[:-1], E_Qinv_B)

        log_Z_pair = 1./2 * logdet_pair
        log_Z_pair -= self.D_latent / 2. * np.log(2 * np.pi)
        log_Z_pair -= 1. / 2 * np.einsum('tij,ti,tj->t', E_BT_Qinv_B,
                                         self.inputs[:-1], self.inputs[:-1])

        return E_AT_Qinv_A, -E_Qinv_A, E_Qinv, h_pair_1, h_pair_2, log_Z_pair

    @property
    def expected_info_emission_params(self):
        # Now get the expected observation potentials
        def get_paramseq(distns):
            contract = partial(np.tensordot, self.expected_states, axes=1)
            params = [d.meanfield_expectedstats() for d in distns]
            return list(map(contract, zip(*params)))

        J_yy, J_yx, J_node, logdet_node = get_paramseq(self.emission_distns)

        n = self.D_latent
        E_Rinv = J_yy
        E_Rinv_C = J_yx[:,:,:n].copy("C")
        E_Rinv_D = J_yx[:,:,n:].copy("C")
        E_DT_Rinv_C = J_node[:,n:,:n]
        E_CT_Rinv_C = J_node[:,:n,:n].copy("C")
        E_DT_Rinv_D = J_node[:,n:,n:]

        h_node = np.einsum('ni,nij->nj', self.data, E_Rinv_C)
        h_node -= np.einsum('ni,nij->nj', self.inputs, E_DT_Rinv_C)

        log_Z_node = -self.D_emission / 2. * np.log(2 * np.pi) * np.ones(self.T)
        log_Z_node += 1. / 2 * logdet_node

        # E[(y-Du)^T R^{-1} (y-Du)]
        log_Z_node -= 1. / 2 * np.einsum('tij,ti,tj->t', E_Rinv,
                                         self.data, self.data)
        log_Z_node -= 1. / 2 * np.einsum('tij,ti,tj->t', -2*E_Rinv_D,
                                         self.data, self.inputs)
        log_Z_node -= 1. / 2 * np.einsum('tij,ti,tj->t', E_DT_Rinv_D,
                                         self.inputs, self.inputs)


        return E_CT_Rinv_C, h_node, log_Z_node

    @property
    def expected_info_params(self):
        return self.expected_info_init_params + \
               self.expected_info_dynamics_params + \
               self.expected_info_emission_params

    @property
    def mf_aBl(self):
        """
        These are the expected log likelihoods (node potentials)
        as seen from the discrete states.
        """
        mf_aBl = self._mf_aBl = np.zeros((self.T, self.num_states))
        ids, dds, eds = self.init_dynamics_distns, self.dynamics_distns, \
            self.emission_distns

        for idx, (d1, d2, d3) in enumerate(zip(ids, dds, eds)):
            mf_aBl[0,idx] = d1.expected_log_likelihood(
                stats=self.E_init_stats)
            mf_aBl[:-1,idx] += d2.expected_log_likelihood(
                stats=self.E_dynamics_stats)
            mf_aBl[:,idx] += d3.expected_log_likelihood(
                stats=self.E_emission_stats)

        mf_aBl[np.isnan(mf_aBl).any(1)] = 0.
        return mf_aBl

    def meanfieldupdate(self, niter=1):
        H_z = self.meanfield_update_discrete_states()
        H_x = self.meanfield_update_gaussian_states()
        self._variational_entropy = H_z + H_x

    def meanfield_update_discrete_states(self):
        super(_SLDSStatesMeanField, self).meanfieldupdate()

        # Compute the variational entropy
        return hmm_entropy(self._mf_param_snapshot, self.all_expected_stats)

    def meanfield_update_gaussian_states(self):
        info_params = self.expected_info_params

        # Call pylds to do message passing and compute expected suff stats
        stats = info_E_step(*self.expected_info_params)
        self._lds_normalizer, self.smoothed_mus, self.smoothed_sigmas, E_xtp1_xtT = stats
        self._set_gaussian_expected_stats(self.smoothed_mus, self.smoothed_sigmas, E_xtp1_xtT)

        # Compute the variational entropy
        return lds_entropy(info_params, stats)

    def get_vlb(self, most_recently_updated=False):
        # E_{q(z)}[log p(z)]
        from pyslds.util import expected_hmm_logprob
        vlb = expected_hmm_logprob(
            self.mf_pi_0, self.mf_trans_matrix,
            (self.expected_states, self.expected_transcounts, self._normalizer))

        # E_{q(x)}[log p(y, x | z)]  is given by aBl
        # To get E_{q(x)}[ aBl ] we multiply and sum
        vlb += np.sum(self.expected_states * self.mf_aBl)

        # Add the variational entropy
        vlb += self._variational_entropy

        # test: compare to old code
        # vlb2 = super(_SLDSStatesMeanField, self).get_vlb(
        #         most_recently_updated=False) \
        #        + self._lds_normalizer
        # print(vlb - vlb2)
        return vlb

    def meanfield_smooth(self):
        # Use the smoothed latent states in combination with the expected
        # discrete states and observation matrices
        expand = lambda a: a[None, ...]
        stack_set = lambda x: np.concatenate(list(map(expand, x)))

        # E_C, E_CCT, E_sigmasq_inv, _ = self.emission_distn.mf_expectations
        if self.model._single_emission:
            # TODO: Improve this
            EC = self.emission_distns[0].mf_expectations[0]
            return self.smoothed_mus.dot(EC.T)
        else:
            # TODO: Improve this
            mf_params = [d.mf_expectations for d in self.emission_distns]
            ECs = stack_set([prms[0] for prms in mf_params])
            ECs = np.tensordot(self.expected_states, ECs, axes=1)
            return np.array([C.dot(mu) for C, mu in zip(ECs, self.smoothed_mus)])



class _SLDSStatesMaskedData(_SLDSStatesGibbs, _SLDSStatesMeanField):
    def __init__(self, model, data=None, mask=None, **kwargs):
        if mask is not None:
            # assert mask.shape == data.shape
            self.mask = mask
        elif data is not None and \
             isinstance(data, np.ndarray) \
             and np.any(np.isnan(data)):

            from warnings import warn
            warn("data includes NaN's. Treating these as missing data.")
            self.mask = ~np.isnan(data)
            # TODO: We should make this unnecessary
            warn("zeroing out nans in data to make sure code works")
            data[np.isnan(data)] = 0
        else:
            self.mask = None

        super(_SLDSStatesMaskedData, self).__init__(model, data=data, **kwargs)

    @property
    def info_emission_params(self):
        if self.mask is None:
            return super(_SLDSStatesMaskedData, self).info_emission_params

        if self.diagonal_noise:
            return self._info_emission_params_diag
        else:
            return self._info_emission_params_dense

    @property
    def _info_emission_params_diag(self):
        if self.model._single_emission:

            C = self.emission_distns[0].A[:,:self.D_latent]
            D = self.emission_distns[0].A[:,self.D_latent:]
            CCT = np.array([np.outer(cp, cp) for cp in C]).\
                reshape((self.D_emission, self.D_latent ** 2))

            sigmasq = self.emission_distns[0].sigmasq_flat
            J_obs = self.mask / sigmasq
            centered_data = self.data - self.inputs.dot(D.T)

            J_node = np.dot(J_obs, CCT)

            # h_node = y^T R^{-1} C - u^T D^T R^{-1} C
            h_node = (centered_data * J_obs).dot(C)

            log_Z_node = -self.mask.sum(1) / 2. * np.log(2 * np.pi)
            log_Z_node -= 1. / 2 * np.sum(self.mask * np.log(sigmasq), axis=1)
            log_Z_node -= 1. / 2 * np.sum(centered_data ** 2 * J_obs, axis=1)


        else:
            expand = lambda a: a[None, ...]
            stack_set = lambda x: np.concatenate(list(map(expand, x)))

            sigmasq_set = [d.sigmasq_flat for d in self.emission_distns]
            sigmasq = stack_set(sigmasq_set)[self.stateseq]
            J_obs = self.mask / sigmasq

            C_set = [d.A[:,:self.D_latent] for d in self.emission_distns]
            D_set = [d.A[:,self.D_latent:] for d in self.emission_distns]
            CCT_set = [np.array([np.outer(cp, cp) for cp in C]).
                           reshape((self.D_emission, self.D_latent**2))
                       for C in C_set]

            J_node = np.zeros((self.T, self.D_latent**2))
            h_node = np.zeros((self.T, self.D_latent))
            log_Z_node = -self.mask.sum(1) / 2. * np.log(2 * np.pi) * np.ones(self.T)

            for i in range(len(self.emission_distns)):
                ti = np.where(self.stateseq == i)[0]
                centered_data_i = self.data[ti] - self.inputs[ti].dot(D_set[i].T)

                J_node[ti] = np.dot(J_obs[ti], CCT_set[i])
                h_node[ti] = (centered_data_i * J_obs[ti]).dot(C_set[i])

                log_Z_node[ti] -= 1. / 2 * np.sum(np.log(sigmasq_set[i]))
                log_Z_node[ti] -= 1. / 2 * np.sum(centered_data_i ** 2 * J_obs[ti], axis=1)

        J_node = J_node.reshape((self.T, self.D_latent, self.D_latent))
        return J_node, h_node, log_Z_node

    @property
    def _info_emission_params_dense(self):
        raise NotImplementedError
        T, D_latent = self.T, self.D_latent
        data, inputs, mask = self.data, self.inputs, self.mask

        Cs, Ds, Rs = self.Cs, self.Ds, self.sigma_obss
        Rinvs = [np.linang.inv(R) for R in Rs]

        # Sloowwwwww
        J_node = np.zeros((T, D_latent, D_latent))
        h_node = np.zeros((T, D_latent, D_latent))
        for t in range(T):
            z = self.stateseq[t]
            Rinv_t = Rinvs[z] * np.outer(mask[t], mask[t])
            J_node[t] = Cs[z].T.dot(Rinv_t).dot(Cs[z])
            h_node[t] = (data[t] - inputs[t].dot(Ds[z].T)).dot(Rinv_t).dot(Cs[z])

        J_node = J_node.reshape((self.T, self.D_latent, self.D_latent))
        return J_node, h_node

    @property
    def expected_info_emission_params(self):
        if self.mask is None:
            return super(_SLDSStatesMaskedData, self).expected_info_emission_params

        if self.model._single_emission:
            # TODO: Fix this! It should use expectations of CDT
            E_CD, E_CDCDT, E_sigmasq_inv, E_log_sigmasq = self.model.emission_distns[0].mf_expectations
            E_C, E_D = E_CD[:, :self.D_latent], E_CD[:, self.D_latent:]
            E_CCT = E_CDCDT[:,:self.D_latent, :self.D_latent]
            E_CDT = E_CDCDT[:,:self.D_latent, self.D_latent:]
            J_obs = self.mask * E_sigmasq_inv

            centered_data = self.data - self.inputs.dot(E_D.T)

            J_node = np.dot(J_obs, E_CCT.reshape((-1, self.D_latent**2))).reshape((-1, self.D_latent, self.D_latent))

            # h_node = y^T R^{-1} C - u^T D^T R^{-1} C
            h_node = (centered_data * J_obs).dot(E_C)

            log_Z_node = -self.mask.sum(1) / 2. * np.log(2 * np.pi)
            log_Z_node -= 1. / 2 * np.sum(self.mask * E_log_sigmasq, axis=1)
            log_Z_node -= 1. / 2 * np.sum(centered_data ** 2 * J_obs, axis=1)

        else:
            raise NotImplementedError("Only supporting diagonal regression class right now")

        return J_node, h_node, log_Z_node

    @property
    def aBl(self):
        if self._aBl is None:
            aBl = self._aBl = np.zeros((self.T, self.num_states))
            ids, dds, eds = self.init_dynamics_distns, self.dynamics_distns, \
                            self.emission_distns

            for idx, (d1, d2) in enumerate(zip(ids, dds)):
                # Initial state distribution
                aBl[0, idx] = d1.log_likelihood(self.gaussian_states[0])

                # Dynamics
                xs = np.hstack((self.gaussian_states[:-1], self.inputs[:-1]))
                aBl[:-1, idx] = d2.log_likelihood((xs, self.gaussian_states[1:]))

            # Emissions
            xs = np.hstack((self.gaussian_states, self.inputs))
            if self.model._single_emission:
                d3 = self.emission_distns[0]
                if self.mask is None:
                    aBl += d3.log_likelihood((xs, self.data))[:,None]
                else:
                    aBl += d3.log_likelihood((xs, self.data), mask=self.mask)[:,None]
            else:
                for idx, d3 in enumerate(eds):
                    if self.mask is None:
                        aBl[:,idx] += d3.log_likelihood((xs, self.data))
                    else:
                        aBl[:,idx] += d3.log_likelihood((xs, self.data), mask=self.mask)

            aBl[np.isnan(aBl).any(1)] = 0.

        return self._aBl

    def _set_gaussian_expected_stats(self, smoothed_mus, smoothed_sigmas, E_xtp1_xtT):
        # TODO: Handle inputs
        if self.mask is None:
            return super(_SLDSStatesMaskedData, self).\
                _set_gaussian_expected_stats(smoothed_mus, smoothed_sigmas, E_xtp1_xtT)

        assert not np.isnan(E_xtp1_xtT).any()
        assert not np.isnan(smoothed_mus).any()
        assert not np.isnan(smoothed_sigmas).any()

        # This is like LDSStates._set_expected_states but doesn't sum over time
        T = self.T
        E_x_xT = smoothed_sigmas + self.smoothed_mus[:, :, None] * self.smoothed_mus[:, None, :]
        E_x_uT = smoothed_mus[:, :, None] * self.inputs[:, None, :]
        E_u_uT = self.inputs[:, :, None] * self.inputs[:, None, :]

        E_xu_xuT = np.concatenate((
            np.concatenate((E_x_xT, E_x_uT), axis=2),
            np.concatenate((np.transpose(E_x_uT, (0, 2, 1)), E_u_uT), axis=2)),
            axis=1)

        E_xut_xutT = E_xu_xuT[:-1]

        E_xtp1_xtp1T = E_x_xT[1:]
        E_xtp1_xtT = E_xtp1_xtT

        E_xtp1_utT = (smoothed_mus[1:, :, None] * self.inputs[:-1, None, :])
        E_xtp1_xutT = np.concatenate((E_xtp1_xtT, E_xtp1_utT), axis=-1)

        # Initial state stats
        self.E_init_stats = (self.smoothed_mus[0], E_x_xT[0], 1.)

        # Dynamics stats
        # TODO avoid memory instantiation by adding to Regression (2TD vs TD^2)
        # TODO only compute EyyT once
        # E_xtp1_xtp1T = self.E_xtp1_xtp1T = E_xu_xuT[1:]
        # E_xt_xtT = self.E_xt_xtT = E_xu_xuT[:-1]
        #
        # self.E_dynamics_stats = \
        #     (E_xtp1_xtp1T, E_xtp1_xtT, E_xt_xtT, np.ones(T-1))
        self.E_dynamics_stats = (E_xtp1_xtp1T, E_xtp1_xutT, E_xut_xutT, np.ones(self.T - 1))

        # Emission stats
        masked_data = self.data * self.mask if self.mask is not None else self.data
        if self.diagonal_noise:
            E_yyT = masked_data ** 2
            E_yxT = masked_data[:, :, None] * self.smoothed_mus[:, None, :]
            E_yuT = masked_data[:, :, None] * self.inputs[:, None, :]
            E_yxuT = np.concatenate((E_yxT, E_yuT), axis=-1)
            self.E_emission_stats = (E_yyT, E_yxuT, E_xu_xuT, self.mask)

        else:
            raise Exception("Only DiagonalRegression currently supports missing data")

        self._mf_aBl = None  # TODO


class _SLDSStatesCountData(_SLDSStatesGibbs):
    def __init__(self, model, data=None, mask=None, **kwargs):
        super(_SLDSStatesCountData, self). \
            __init__(model, data=data, mask=mask, **kwargs)

        # Check if the emission matrix is a count regression
        if isinstance(self.emission_distns[0], _PGLogisticRegressionBase):
            self.has_count_data = True

            # Initialize the Polya-gamma samplers
            num_threads = ppg.get_omp_num_threads()
            seeds = np.random.randint(2 ** 16, size=num_threads)
            self.ppgs = [ppg.PyPolyaGamma(seed) for seed in seeds]

            # Initialize auxiliary variables, omega
            self.omega = np.ones((self.T, self.D_emission), dtype=np.float)
        else:
            self.has_count_data = False

    @property
    def sigma_obss(self):
        if self.has_count_data:
            raise Exception("Count data does not have sigma_obs")
        return super(_SLDSStatesCountData, self).sigma_obss

    @property
    def info_emission_params(self):
        if not self.has_count_data:
            return super(_SLDSStatesCountData, self).info_emission_params

        # Otherwise, use the Polya-gamma augmentation
        # log p(y_{tn} | x, om)
        #   = -0.5 * om_{tn} * (c_n^T x_t + d_n^T u_t + b_n)**2
        #     + kappa * (c_n * x_t + d_n^Tu_t + b_n)
        #   = -0.5 * om_{tn} * (x_t^T c_n c_n^T x_t
        #                       + 2 x_t^T c_n d_n^T u_t
        #                       + 2 x_t^T c_n b_n)
        #     + x_t^T (kappa_{tn} * c_n)
        #   = -0.5 x_t^T (c_n c_n^T * om_{tn}) x_t
        #     +  x_t^T * (kappa_{tn} - d_n^T u_t * om_{tn} -b_n * om_{tn}) * c_n
        #
        # Thus
        # J = (om * mask).dot(CCT)
        # h = ((kappa - om * d) * mask).dot(C)
        T, D_latent, D_emission = self.T, self.D_latent, self.D_emission
        data, inputs, mask, omega = self.data, self.inputs, self.mask, self.omega
        if self.model._single_emission:
            emission_distn = self.emission_distns[0]
            C = emission_distn.A[:, :D_latent]
            D = emission_distn.A[:,D_latent:]
            b = emission_distn.b
            CCT = np.array([np.outer(cp, cp) for cp in C]).\
                reshape((D_emission, D_latent ** 2))

            J_node = np.dot(omega * mask, CCT)
            J_node = J_node.reshape((T, D_latent, D_latent))

            kappa = emission_distn.kappa_func(data)
            h_node = ((kappa - omega * b.T - omega * inputs.dot(D.T)) * mask).dot(C)

        else:
            C_set = [d.A[:,:self.D_latent] for d in self.emission_distns]
            D_set = [d.A[:,self.D_latent:] for d in self.emission_distns]
            b_set = [d.b for d in self.emission_distns]
            CCT_set = [np.array([np.outer(cp, cp) for cp in C]).
                           reshape((self.D_emission, self.D_latent**2))
                       for C in C_set]

            J_node = np.zeros((self.T, self.D_latent**2))
            h_node = np.zeros((self.T, self.D_latent))

            for i in range(len(self.emission_distns)):
                ti = np.where(self.stateseq == i)[0]
                J_obs = omega[ti] * mask[ti]
                kappa = self.emission_distns[i].kappa_func(data[ti])

                J_node[ti] = np.dot(J_obs, CCT_set[i])

                h_node[ti] = ((kappa
                               - omega[ti] * b_set[i].T
                               - omega * inputs[ti].dot(D_set[i].T)
                               ) * mask[ti]).dot(C_set[i])

        # See pylds/states.py for info on the log normalizer
        # terms for Polya-gamma augmented states

        return J_node, h_node, np.zeros(self.T)

    @property
    def expected_info_emission_params(self):
        if self.has_count_data:
            raise NotImplementedError("Mean field with count observations is not yet supported")

        return super(_SLDSStatesCountData, self).expected_info_emission_params

    def log_likelihood(self):
        if self.has_count_data:

            if self.model._single_emission:
                ll = self.emission_distns[0].log_likelihood(
                    (np.hstack((self.gaussian_states, self.inputs)),
                     self.data), mask=self.mask).sum()
            else:
                ll = 0
                z, xs, u, y = self.stateseq, self.gaussian_states, self.inputs, self.data
                for k, ed in enumerate(self.emission_distns):
                    xuk = np.hstack((xs[z==k], u[z==k]))
                    yk = y[z==k]
                    ll += ed.log_likelihood((xuk, yk), mask=self.mask).sum()
            return ll

        else:
            return super(_SLDSStatesCountData, self).log_likelihood()

    @staticmethod
    def empirical_rate(data, sigma=3.0):
        """
        Smooth count data to get an empirical rate
        """
        from scipy.ndimage.filters import gaussian_filter1d
        return 0.001 + gaussian_filter1d(data.astype(np.float), sigma, axis=0)

    def resample(self, niter=1):
        niter = self.niter if hasattr(self, 'niter') else niter
        for itr in range(niter):
            self.resample_discrete_states()
            self.resample_gaussian_states()

            if self.has_count_data:
                self.resample_auxiliary_variables()

    def resample_auxiliary_variables(self):
        if self.model._single_emission:
            ed = self.emission_distns[0]
            C, D = ed.A[:, :self.D_latent], ed.A[:, self.D_latent:]
            psi = self.gaussian_states.dot(C.T) + self.inputs.dot(D.T) + ed.b.T
            b = ed.b_func(self.data)
        else:
            C_set = [d.A[:, :self.D_latent] for d in self.emission_distns]
            D_set = [d.A[:, self.D_latent:] for d in self.emission_distns]
            b_set = [d.b for d in self.emission_distns]

            psi = np.zeros((self.T, self.D_emission))
            b = np.zeros((self.T, self.D_emission))

            for i in range(len(self.emission_distns)):
                ti = np.where(self.stateseq == i)[0]
                psi[ti] = self.gaussian_states[ti].dot(C_set[i].T)
                psi[ti] += self.inputs[ti].dot(D_set[i].T)
                psi[ti] += b_set[i].T

                b[ti] = self.emission_distns[i].b_func(self.data[ti])

        ppg.pgdrawvpar(self.ppgs, b.ravel(), psi.ravel(), self.omega.ravel())

    def smooth(self):
        if not self.has_count_data:
            return super(_SLDSStatesCountData, self).smooth()

        X = np.column_stack((self.gaussian_states, self.inputs))
        if self.model._single_emission:
            ed = self.emission_distns[0]
            mean = ed.mean(X)

        else:
            mean = np.zeros((self.T, self.D_emission))
            for i, ed in enumerate(self.emission_distns):
                ed = self.emission_distns[i]
                ti = np.where(self.stateseq == i)[0]
                mean[ti] = ed.mean(X[ti])

        return mean


####################
#  states classes  #
####################

class HMMSLDSStatesPython(
    _SLDSStatesCountData,
    _SLDSStatesMaskedData,
    _SLDSStatesGibbs,
    _SLDSStatesVBEM,
    _SLDSStatesMeanField,
    HMMStatesPython):
    pass


class HMMSLDSStatesEigen(
    _SLDSStatesCountData,
    _SLDSStatesMaskedData,
    _SLDSStatesGibbs,
    _SLDSStatesVBEM,
    _SLDSStatesMeanField,
    HMMStatesEigen):
    pass


class HSMMSLDSStatesPython(
    _SLDSStatesCountData,
    _SLDSStatesMaskedData,
    _SLDSStatesGibbs,
    _SLDSStatesVBEM,
    _SLDSStatesMeanField,
    HSMMStatesPython):
    pass


class HSMMSLDSStatesEigen(
    _SLDSStatesCountData,
    _SLDSStatesMaskedData,
    _SLDSStatesGibbs,
    _SLDSStatesVBEM,
    _SLDSStatesMeanField,
    HSMMStatesEigen):
    pass


class GeoHSMMSLDSStates(
    _SLDSStatesCountData,
    _SLDSStatesMaskedData,
    _SLDSStatesGibbs,
    _SLDSStatesVBEM,
    _SLDSStatesMeanField,
    GeoHSMMStates):
    pass
