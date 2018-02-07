from __future__ import division
import numpy as np
from functools import partial

from pyhsmm.internals.hmm_states import HMMStatesPython, HMMStatesEigen
from pyhsmm.internals.hsmm_states import HSMMStatesPython, HSMMStatesEigen, \
    GeoHSMMStates

from pylds.lds_messages_interface import info_E_step, info_sample

from pyslds.util import hmm_entropy, lds_entropy, expected_regression_log_prob, expected_gaussian_logprob

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
            self.stateseq = np.array(stateseq, dtype=np.int32)

        elif generate:
            self.generate_states(stateseq=stateseq)

        if data is not None and not initialize_from_prior:
            self.resample()

    def generate_states(self, initial_condition=None, with_noise=True, stateseq=None):
        """
        Jointly sample the discrete and continuous states
        """
        from pybasicbayes.util.stats import sample_discrete
        # Generate from the prior and raise exception if unstable
        T, K, n = self.T, self.num_states, self.D_latent
        A = self.trans_matrix

        # Initialize discrete state sequence
        dss = -1 * np.ones(T, dtype=np.int32) if stateseq is None else stateseq.astype(np.int32)
        assert dss.shape == (T,)
        gss = np.empty((T,n), dtype='double')

        if initial_condition is None:
            if dss[0] == -1:
                dss[0] = sample_discrete(self.pi_0)
            gss[0] = self.init_dynamics_distns[dss[0]].rvs()
        else:
            dss[0] = initial_condition[0]
            gss[0] = initial_condition[1]

        for t in range(1,T):
            # Sample discrete state given previous continuous state
            if with_noise:
                # Sample discre=te state from recurrent transition matrix
                if dss[t] == -1:
                    dss[t] = sample_discrete(A[dss[t-1], :])

                # Sample continuous state given current discrete state
                gss[t] = self.dynamics_distns[dss[t-1]].\
                    rvs(x=np.hstack((gss[t-1][None,:], self.inputs[t-1][None,:])),
                        return_xy=False)
            else:
                # Pick the most likely next discrete state and continuous state
                if dss[t] == -1:
                    dss[t] = np.argmax(A[dss[t-1], :])

                gss[t] = self.dynamics_distns[dss[t-1]]. \
                    predict(np.hstack((gss[t-1][None,:], self.inputs[t-1][None,:])))
            assert np.all(np.isfinite(gss[t])), "SLDS appears to be unstable!"

        self.stateseq = dss
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
    def single_emission(self):
        return self.model._single_emission

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
    def A_set(self):
        return np.concatenate([d.A[None, :, :self.D_latent] for d in self.dynamics_distns])

    @property
    def As(self):
        return self.A_set[self.stateseq]

    @property
    def B_set(self):
        return np.concatenate([d.A[None, :, self.D_latent:] for d in self.dynamics_distns])

    @property
    def Bs(self):
        return self.B_set[self.stateseq]

    @property
    def Q_set(self):
        return np.concatenate([d.sigma[None,...] for d in self.dynamics_distns])

    @property
    def sigma_statess(self):
        return self.Q_set[self.stateseq]

    @property
    def C_set(self):
        return np.concatenate([d.A[None,:,:self.D_latent] for d in self.emission_distns])

    @property
    def Cs(self):
        return self.C_set[self.stateseq]

    @property
    def D_set(self):
        return np.concatenate([d.A[None, :, self.D_latent:] for d in self.emission_distns])

    @property
    def Ds(self):
        return self.D_set[self.stateseq]

    @property
    def R_set(self):
        return np.concatenate([d.sigma[None,...] for d in self.emission_distns])

    @property
    def Rinv_set(self):
        if self.diagonal_noise:
            return np.concatenate([np.diag(1. / d.sigmasq_flat)[None,...] for d in self.emission_distns])
        else:
            return np.concatenate([np.linalg.inv(d.sigma)[None,...] for d in self.emission_distns])

    @property
    def sigma_obss(self):
        return self.R_set[self.stateseq]

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

        A_set, B_set, Q_set = self.A_set, self.B_set, self.Q_set

        # Get the pairwise potentials
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

        C_set, D_set, Ri_set = self.C_set, self.D_set, self.Rinv_set
        RiC_set = [Ri.dot(C) for C,Ri in zip(C_set, Ri_set)]
        RiD_set = [Ri.dot(D) for D,Ri in zip(D_set, Ri_set)]
        CRiC_set = [C.T.dot(RiC) for C,RiC in zip(C_set, RiC_set)]
        DRiC_set = [D.T.dot(RiC) for D,RiC in zip(D_set, RiC_set)]
        DRiD_set = [D.T.dot(RiD) for D,RiD in zip(D_set, RiD_set)]

        if self.single_emission:
            Ri = Ri_set[0]
            RiC = RiC_set[0]
            RiD = RiD_set[0]
            DRiC = DRiC_set[0]
            DRiD = DRiD_set[0]
            J_node = CRiC_set[0]
            h_node = np.dot(self.data, RiC)
            h_node -= np.dot(self.inputs, DRiC)
            logdet = np.linalg.slogdet(Ri_set[0])[1]

            log_Z_node = -self.D_emission / 2. * np.log(2 * np.pi) * np.ones(self.T)
            log_Z_node += 1. / 2 * logdet

            # E[(y-Du)^T R^{-1} (y-Du)]
            log_Z_node -= 1. / 2 * np.sum(np.dot(self.data, Ri) * self.data, axis=1)
            log_Z_node += np.sum(np.dot(self.data, RiD) * self.inputs, axis=1)
            log_Z_node -= 1. / 2 * np.sum(np.dot(self.inputs, DRiD) * self.inputs, axis=1)

        else:
            Ri = stack_set(Ri_set)[self.stateseq]
            RiC = stack_set(RiC_set)[self.stateseq]
            RiD = stack_set(RiD_set)[self.stateseq]
            DRiC = stack_set(DRiC_set)[self.stateseq]
            DRiD = stack_set(DRiD_set)[self.stateseq]
            J_node = stack_set(CRiC_set)[self.stateseq]
            h_node = np.einsum('ni,nij->nj', self.data, RiC)
            h_node -= np.einsum('ni,nij->nj', self.inputs, DRiC)
            logdet = stack_set([np.linalg.slogdet(R)[1] for R in Ri_set])[self.stateseq]

            log_Z_node = -self.D_emission / 2. * np.log(2 * np.pi) * np.ones(self.T)
            log_Z_node += 1. / 2 * logdet

            # E[(y-Du)^T R^{-1} (y-Du)]
            log_Z_node -= 1. / 2 * np.einsum('tij,ti,tj->t', Ri, self.data, self.data)
            log_Z_node -= 1. / 2 * np.einsum('tij,ti,tj->t', -2 * RiD, self.data, self.inputs)
            log_Z_node -= 1. / 2 * np.einsum('tij,ti,tj->t', DRiD, self.inputs, self.inputs)

        return J_node, h_node, log_Z_node

    @property
    def info_params(self):
        return self.info_init_params + self.info_dynamics_params + self.info_emission_params

    @property
    def aBl(self):
        if self._aBl is None:
            aBl = self._aBl = np.zeros((self.T, self.num_states))
            ids, dds, eds = self.init_dynamics_distns, self.dynamics_distns, self.emission_distns

            for idx, (d1, d2) in enumerate(zip(ids, dds)):
                # Initial state distribution
                aBl[0, idx] = d1.log_likelihood(self.gaussian_states[0])

                # Dynamics
                xs = np.hstack((self.gaussian_states[:-1], self.inputs[:-1]))
                aBl[:-1, idx] = d2.log_likelihood((xs, self.gaussian_states[1:]))

            # Emissions
            xs = np.hstack((self.gaussian_states, self.inputs))
            if self.single_emission:
                d3 = self.emission_distns[0]
                aBl += d3.log_likelihood((xs, self.data))[:, None]

            else:
                for idx, d3 in enumerate(eds):
                    aBl[:, idx] += d3.log_likelihood((xs, self.data))

            aBl[np.isnan(aBl).any(1)] = 0.

        return self._aBl

    ### Mean-field and VBEM base functions.
    def smooth(self):
        # Use the info E step because it can take advantage of diagonal noise
        # The standard E step could but we have not implemented it
        self.info_E_step()
        xu = np.column_stack((self.smoothed_mus, self.inputs))
        if self.single_emission:
            return xu.dot(self.emission_distns[0].A.T)
        else:
            return np.array([C.dot(x) + D.dot(u) for C, D, x, u in
                             zip(self.Cs, self.Ds, self.smoothed_mus, self.inputs)])

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
        T, D_obs = self.T, self.D_emission
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

        # Emission stats -- special case diagonal noise
        E_yyT = self.data**2 if self.diagonal_noise else self.data[:, :, None] * self.data[:, None, :]
        E_yxT = self.data[:, :, None] * self.smoothed_mus[:, None, :]
        E_yuT = self.data[:, :, None] * self.inputs[:, None, :]
        E_yxuT = np.concatenate((E_yxT, E_yuT), axis=-1)
        E_n = np.ones((T, D_obs)) if self.diagonal_noise else np.ones(T)
        self.E_emission_stats = (E_yyT, E_yxuT, E_xu_xuT, E_n)


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
        raise NotImplementedError

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

        A_set, B_set, Q_set = self.A_set, self.B_set, self.Q_set

        # Get the pairwise potentials
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

        C_set, D_set, Ri_set = self.C_set, self.D_set, self.Rinv_set
        RiC_set = stack_set([Ri.dot(C) for C, Ri in zip(C_set, Ri_set)])
        RiD_set = stack_set([Ri.dot(D) for D, Ri in zip(D_set, Ri_set)])
        CRiC_set = stack_set([C.T.dot(RiC) for C, RiC in zip(C_set, RiC_set)])
        DRiC_set = stack_set([D.T.dot(RiC) for D, RiC in zip(D_set, RiC_set)])
        DRiD_set = stack_set([D.T.dot(RiD) for D, RiD in zip(D_set, RiD_set)])

        if self.single_emission:
            Ri = Ri_set[0]
            RiC = RiC_set[0]
            RiD = RiD_set[0]
            DRiC = DRiC_set[0]
            DRiD = DRiD_set[0]
            logdet = np.linalg.slogdet(Ri_set[0])[1]

            J_node = CRiC_set[0]
            h_node = np.dot(self.data, RiC)
            h_node -= np.dot(self.inputs, DRiC)

            log_Z_node = -self.D_emission / 2. * np.log(2 * np.pi) * np.ones(self.T)
            log_Z_node += 1. / 2 * logdet
            log_Z_node -= 1. / 2 * np.sum(np.dot(self.data, Ri) * self.data, axis=1)
            log_Z_node += np.sum(np.dot(self.data, RiD) * self.inputs, axis=1)
            log_Z_node -= 1. / 2 * np.sum(np.dot(self.inputs, DRiD) * self.inputs, axis=1)

        else:
            E_z = self.expected_states
            Ri = np.tensordot(E_z, Ri_set, axes=1)
            RiC = np.tensordot(E_z, RiC_set, axes=1)
            RiD = np.tensordot(E_z, RiD_set, axes=1)
            DRiC = np.tensordot(E_z, DRiC_set, axes=1)
            DRiD = np.tensordot(E_z, DRiD_set, axes=1)
            logdet = np.dot(E_z, np.array([np.linalg.slogdet(Ri)[1] for Ri in Ri_set]))

            J_node = np.tensordot(E_z, CRiC_set, axes=1)
            h_node = np.einsum('ni,nij->nj', self.data, RiC)
            h_node -= np.einsum('ni,nij->nj', self.inputs, DRiC)

            log_Z_node = -self.D_emission / 2. * np.log(2 * np.pi) * np.ones(self.T)
            log_Z_node += 1. / 2 * logdet
            log_Z_node -= 1. / 2 * np.einsum('tij,ti,tj->t', Ri, self.data, self.data)
            log_Z_node -= 1. / 2 * np.einsum('tij,ti,tj->t', -2 * RiD, self.data, self.inputs)
            log_Z_node -= 1. / 2 * np.einsum('tij,ti,tj->t', DRiD, self.inputs, self.inputs)

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
        ids, dds, eds = self.init_dynamics_distns, self.dynamics_distns, self.emission_distns

        for k, (id, dd) in enumerate(zip(ids, dds)):
            vbem_aBl[0, k] = expected_gaussian_logprob(id.mu, id.sigma, self.E_init_stats)
            vbem_aBl[:-1, k] += expected_regression_log_prob(dd, self.E_dynamics_stats)

        if self.single_emission:
            ed = self.emission_distns[0]
            vbem_aBl += expected_regression_log_prob(ed, self.E_emission_stats)[:,None]
        else:
            for k, ed in enumerate(self.emission_distns):
                vbem_aBl[:, k] += expected_regression_log_prob(ed, self.E_emission_stats)

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
        self._aBl = 0

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

        # Save the states
        self.stateseq = np.argmax(self.expected_states, axis=1)

        # Compute the variational entropy
        return hmm_entropy(self._mf_param_snapshot, self.all_expected_stats)

    def meanfield_update_gaussian_states(self):
        info_params = self.expected_info_params

        # Call pylds to do message passing and compute expected suff stats
        stats = info_E_step(*self.expected_info_params)
        self._lds_normalizer, self.smoothed_mus, self.smoothed_sigmas, E_xtp1_xtT = stats
        self._set_gaussian_expected_stats(self.smoothed_mus, self.smoothed_sigmas, E_xtp1_xtT)

        # Save the states
        self.gaussian_states = self.smoothed_mus.copy()

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

        if self.single_emission:
            EC = self.emission_distns[0].mf_expectations[0]
            return self.smoothed_mus.dot(EC.T)
        else:
            mf_params = [d.mf_expectations for d in self.emission_distns]
            ECs = stack_set([prms[0] for prms in mf_params])
            ECs = np.tensordot(self.expected_states, ECs, axes=1)
            return np.array([C.dot(mu) for C, mu in zip(ECs, self.smoothed_mus)])


class _SLDSStatesMaskedData(_SLDSStatesGibbs, _SLDSStatesVBEM, _SLDSStatesMeanField):
    """
    This mixin allows arbitrary patterns of missing data.  Currently,
    we only support the simplest case in which the observation noise
    has diagonal covariance, such that,

        y_{t,n} ~ N(c_n \dot x_t,  \sigma^2_n).

    In this case, missing data corresponds to fewer emission potentials.

    The missing data can either be indicated by NaN's in the data or by
    an explicit, Boolean mask passed to the constructor.  The mixin works
    by overriding the info_emission_parameters and the corresponding
    emission likelihoods.  If no mask is present, it passes through
    to the base mixings (Gibbs, VBEM, MeanField).
    """
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
            data[np.isnan(data)] = 0
        else:
            self.mask = None

        super(_SLDSStatesMaskedData, self).__init__(model, data=data, **kwargs)

        # If masked, make sure we have diagonal observations
        # we do not currently support arbitrary masking with dense observation cov.
        if self.mask is not None and not self.diagonal_noise:
            raise Exception("PySLDS only supports diagonal observation noise with masked data")

    def heldout_log_likelihood(self, test_mask=None):
        """
        Compute the log likelihood of the masked data given the latent
        discrete and continuous states.
        """
        if test_mask is None:
            # If a test mask is not supplied, use the negation of this object's mask
            if self.mask is None:
                return 0
            else:
                test_mask = ~self.mask

        xs = np.hstack((self.gaussian_states, self.inputs))
        if self.single_emission:
            return self.emission_distns[0].\
                log_likelihood((xs, self.data), mask=test_mask).sum()
        else:
            hll = 0
            z = self.stateseq
            for idx, ed in enumerate(self.emission_distns):
                hll += ed.log_likelihood((xs[z == idx], self.data[z == idx]),
                                         mask=test_mask[z == idx]).sum()

    ### Gibbs
    @property
    def info_emission_params(self):
        if self.mask is None:
            return super(_SLDSStatesMaskedData, self).info_emission_params

        # Otherwise, compute masked potentials
        expand = lambda a: a[None, ...]
        stack_set = lambda x: np.concatenate(list(map(expand, x)))

        C_set, D_set = self.C_set, self.D_set
        sigmasq_inv_set = [np.diag(Ri) for Ri in self.Rinv_set]
        CCT_set = stack_set(
            [np.array([np.outer(cp, cp) for cp in C]).
                 reshape((self.D_emission, self.D_latent ** 2)) for C in C_set])

        # Compute expectations wrt q(z)
        if self.single_emission:
            sigmasq_inv = sigmasq_inv_set[0]
            C = C_set[0]
            D = D_set[0]
            CCT = CCT_set[0]
        else:
            z = self.stateseq
            sigmasq_inv = sigmasq_inv_set[z]
            C = C_set[z]
            D = D_set[z]
            CCT = CCT_set[z]

        # Finally, we can compute the emission potential with the mask
        T, D_latent, data, inputs, mask = self.T, self.D_latent, self.data, self.inputs, self.mask
        centered_data = data - inputs.dot(np.swapaxes(D, -2, -1))
        J_node = np.dot(mask * sigmasq_inv, CCT).reshape((T, D_latent, D_latent))
        h_node = (mask * centered_data * sigmasq_inv).dot(C)

        log_Z_node = -mask.sum(1) / 2. * np.log(2 * np.pi) * np.ones(T)
        log_Z_node += 1. / 2 * np.sum(mask * np.log(sigmasq_inv))
        log_Z_node += -1. / 2 * np.sum(mask * centered_data ** 2 * sigmasq_inv, axis=1)

        return J_node, h_node, log_Z_node

    @property
    def aBl(self):
        if self.mask is None:
            return super(_SLDSStatesMaskedData, self).aBl

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
            if self.single_emission:
                d3 = self.emission_distns[0]
                aBl += d3.log_likelihood((xs, self.data), mask=self.mask)[:, None]
            else:
                for idx, d3 in enumerate(eds):
                    aBl[:, idx] += d3.log_likelihood((xs, self.data), mask=self.mask)

            aBl[np.isnan(aBl).any(1)] = 0.

        return self._aBl

    ### VBEM
    @property
    def vbem_info_emission_params(self):
        if self.mask is None:
            return super(_SLDSStatesMaskedData, self).vbem_info_emission_params

        # Otherwise, compute masked potentials
        expand = lambda a: a[None, ...]
        stack_set = lambda x: np.concatenate(list(map(expand, x)))

        C_set, D_set = self.C_set, self.D_set
        sigmasq_inv_set = [np.diag(Ri) for Ri in self.Rinv_set]
        CCT_set = stack_set(
            [np.array([np.outer(cp, cp) for cp in C]).
                 reshape((self.D_emission, self.D_latent ** 2)) for C in C_set])

        # Compute expectations wrt q(z)
        if self.single_emission:
            sigmasq_inv = sigmasq_inv_set[0]
            C = C_set[0]
            D = D_set[0]
            CCT = CCT_set[0]
        else:
            E_z = self.expected_states
            sigmasq_inv = np.tensordot(E_z, sigmasq_inv_set, axes=1)
            C = np.tensordot(E_z, C_set, axes=1)
            D = np.tensordot(E_z, D_set, axes=1)
            CCT = np.tensordot(E_z, CCT_set, axes=1)

        # Finally, we can compute the emission potential with the mask
        T, D_latent, data, inputs, mask = self.T, self.D_latent, self.data, self.inputs, self.mask
        centered_data = data - inputs.dot(np.swapaxes(D, -2, -1))
        J_node = np.dot(mask * sigmasq_inv, CCT).reshape((T, D_latent, D_latent))
        h_node = (mask * centered_data * sigmasq_inv).dot(C)

        log_Z_node = -mask.sum(1) / 2. * np.log(2 * np.pi)
        log_Z_node += 1. / 2 * np.sum(mask * np.log(sigmasq_inv), axis=1)
        log_Z_node += -1. / 2 * np.sum(mask * centered_data ** 2 * sigmasq_inv, axis=1)

        return J_node, h_node, log_Z_node

    ### Mean field
    @property
    def expected_info_emission_params(self):
        if self.mask is None:
            return super(_SLDSStatesMaskedData, self).expected_info_emission_params

        # Otherwise, compute masked potentials
        expand = lambda a: a[None, ...]
        stack_set = lambda x: np.concatenate(list(map(expand, x)))

        # mf_expectations: mf_E_A, mf_E_AAT, mf_E_sigmasq_inv, mf_E_log_sigmasq
        mf_stats = [ed.mf_expectations for ed in self.emission_distns]

        n = self.D_latent
        E_C_set = stack_set([s[0][:, :n] for s in mf_stats])
        E_D_set = stack_set([s[0][:, n:] for s in mf_stats])
        E_CCT_set = stack_set([s[1][:, :n, :n] for s in mf_stats])
        E_sigmasq_inv_set = stack_set(s[2] for s in mf_stats)
        E_log_sigmasq_inv_set = stack_set(s[3] for s in mf_stats)

        # Compute expectations wrt q(z)
        if self.single_emission:
            E_C = E_C_set[0]
            E_D = E_D_set[0]
            E_CCT = E_CCT_set[0]
            E_sigmasq_inv = E_sigmasq_inv_set[0]
            E_log_sigmasq_inv = E_log_sigmasq_inv_set[0]
        else:
            E_z = self.expected_states
            E_C = np.tensordot(E_z, E_C_set, axes=1)
            E_D = np.tensordot(E_z, E_D_set, axes=1)
            E_CCT = np.tensordot(E_z, E_CCT_set, axes=1)
            E_sigmasq_inv = np.tensordot(E_z, E_sigmasq_inv_set, axes=1)
            E_log_sigmasq_inv = np.tensordot(E_z, E_log_sigmasq_inv_set, axes=1)

        # Finally, we can compute the emission potential with the mask
        T, D_latent, data, inputs, mask = self.T, self.D_latent, self.data, self.inputs, self.mask
        centered_data = data - inputs.dot(np.swapaxes(E_D, -2, -1))
        J_node = np.tensordot(mask * E_sigmasq_inv, E_CCT, axes=1).\
            reshape((T, D_latent, D_latent))
        h_node = (mask * centered_data * E_sigmasq_inv).dot(E_C)

        log_Z_node = -mask.sum(1) / 2. * np.log(2 * np.pi)
        log_Z_node += 1. / 2 * np.sum(mask * E_log_sigmasq_inv, axis=1)
        log_Z_node += -1. / 2 * np.sum(mask * centered_data ** 2 * E_sigmasq_inv, axis=1)

        return J_node, h_node, log_Z_node

    ### VBEM and Mean Field
    def _set_gaussian_expected_stats(self, smoothed_mus, smoothed_sigmas, E_xtp1_xtT):
        if self.mask is None:
            return super(_SLDSStatesMaskedData, self). \
                _set_gaussian_expected_stats(smoothed_mus, smoothed_sigmas, E_xtp1_xtT)

        assert not np.isnan(E_xtp1_xtT).any()
        assert not np.isnan(smoothed_mus).any()
        assert not np.isnan(smoothed_sigmas).any()
        assert smoothed_mus.shape == (self.T, self.D_latent)
        assert smoothed_sigmas.shape == (self.T, self.D_latent, self.D_latent)
        assert E_xtp1_xtT.shape == (self.T - 1, self.D_latent, self.D_latent)

        # This is like LDSStates._set_expected_states but doesn't sum over time
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
        self.E_dynamics_stats = (E_xtp1_xtp1T, E_xtp1_xutT, E_xut_xutT, np.ones(self.T - 1))

        # Emission stats
        masked_data = self.data * self.mask
        E_yyT = masked_data ** 2
        E_yxT = masked_data[:, :, None] * self.smoothed_mus[:, None, :]
        E_yuT = masked_data[:, :, None] * self.inputs[:, None, :]
        E_yxuT = np.concatenate((E_yxT, E_yuT), axis=-1)

        # We can't just reuse E[xu \dot xu^T].  Now we need to reweight by mask.
        # Now E[xu \dot xu^T] must be T x D_obs x (D_latent + D_input) x (D_latent + D_input)
        E_xu_xuT_masked = self.mask[:, :, None, None] * E_xu_xuT[:, None, :, :]
        E_n = self.mask.astype(np.float)
        self.E_emission_stats = (E_yyT, E_yxuT, E_xu_xuT_masked, E_n)


####################
#  states classes  #
####################

class HMMSLDSStatesPython(
    _SLDSStatesMaskedData,
    _SLDSStatesGibbs,
    _SLDSStatesVBEM,
    _SLDSStatesMeanField,
    HMMStatesPython):
    pass


class HMMSLDSStatesEigen(
    _SLDSStatesMaskedData,
    _SLDSStatesGibbs,
    _SLDSStatesVBEM,
    _SLDSStatesMeanField,
    HMMStatesEigen):
    pass


class HSMMSLDSStatesPython(
    _SLDSStatesMaskedData,
    _SLDSStatesGibbs,
    _SLDSStatesVBEM,
    _SLDSStatesMeanField,
    HSMMStatesPython):
    pass


class HSMMSLDSStatesEigen(
    _SLDSStatesMaskedData,
    _SLDSStatesGibbs,
    _SLDSStatesVBEM,
    _SLDSStatesMeanField,
    HSMMStatesEigen):
    pass


class GeoHSMMSLDSStates(
    _SLDSStatesMaskedData,
    _SLDSStatesGibbs,
    _SLDSStatesVBEM,
    _SLDSStatesMeanField,
    GeoHSMMStates):
    pass


class _SLDSStatesCountData(_SLDSStatesMaskedData, _SLDSStatesGibbs):
    def __init__(self, model, data=None, mask=None, **kwargs):
        super(_SLDSStatesCountData, self). \
            __init__(model, data=data, mask=mask, **kwargs)

        # Check if the emission matrix is a count regression
        import pypolyagamma as ppg
        from pypolyagamma.distributions import _PGLogisticRegressionBase

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
    def diagonal_noise(self):
        from pypolyagamma.distributions import _PGLogisticRegressionBase
        from pybasicbayes.distributions import DiagonalRegression
        return all([isinstance(ed, (_PGLogisticRegressionBase, DiagonalRegression))
                    for ed in self.emission_distns])

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
        if self.single_emission:
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

    def log_likelihood(self):
        if self.has_count_data:

            if self.single_emission:
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
        import pypolyagamma as ppg
        if self.single_emission:
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
        if self.single_emission:
            ed = self.emission_distns[0]
            mean = ed.mean(X)

        else:
            mean = np.zeros((self.T, self.D_emission))
            for i, ed in enumerate(self.emission_distns):
                ed = self.emission_distns[i]
                ti = np.where(self.stateseq == i)[0]
                mean[ti] = ed.mean(X[ti])

        return mean

    ### VBEM
    @property
    def vbem_info_emission_params(self):
        raise NotImplementedError("VBEM not implemented for Polya-gamma augmented states.")

    def vb_E_step(self):
        raise NotImplementedError("VBEM not implemented for Polya-gamma augmented states.")



class HMMCountSLDSStatesPython(
    _SLDSStatesCountData,
    HMMStatesPython):
    pass


class HMMCountSLDSStatesEigen(
    _SLDSStatesCountData,
    HMMStatesEigen):
    pass


class HSMMCountSLDSStatesPython(
    _SLDSStatesCountData,
    HSMMStatesPython):
    pass


class HSMMCountSLDSStatesEigen(
    _SLDSStatesCountData,
    HSMMStatesEigen):
    pass


class GeoHSMMCountSLDSStates(
    _SLDSStatesCountData,
    GeoHSMMStates):
    pass
