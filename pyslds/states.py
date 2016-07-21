from __future__ import division
import numpy as np
from functools import partial

from pybasicbayes.util.stats import mniw_expectedstats

from pyhsmm.internals.hmm_states import HMMStatesPython, HMMStatesEigen, _StatesBase
from pyhsmm.internals.hsmm_states import HSMMStatesPython, HSMMStatesEigen, \
    GeoHSMMStates

from autoregressive.util import AR_striding
from pylds.states import LDSStates
from pylds.lds_messages_interface import filter_and_sample, info_E_step, info_sample

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
        elif generate:
            if data is not None and not initialize_from_prior:
                self.resample()
            else:
                self.generate_states()

    def generate_states(self):
        super(_SLDSStates,self).generate_states()
        self.generate_gaussian_states()

    def generate_gaussian_states(self):
        # TODO: Handle inputs
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
        # TODO: Handle inputs
        # Go through each time bin, get the discrete latent state,
        # use that to index into the emission_distns to get samples
        T, p = self.T, self.D_emission
        dss, gss = self.stateseq, self.gaussian_states
        data = np.empty((T,p),dtype='double')

        for t in range(self.T):
            data[t] = self.emission_distns[dss[t]].\
                rvs(x=np.hstack((gss[t][None, :], self.inputs[t][None,:])),
                    return_xy=False)

        return data

    ## convenience properties

    @property
    def D_latent(self):
        return self.dynamics_distns[0].D_out

    @property
    def D_input(self):
        return self.dynamics_distns[0].D_out - self.dynamics_distns[0].D_in

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
        return J_init, h_init

    @property
    def info_dynamics_params(self):
        expand = lambda a: a[None,...]
        stack_set = lambda x: np.concatenate(map(expand, x))

        A_set = [d.A for d in self.dynamics_distns]
        Q_set = [d.sigma for d in self.dynamics_distns]

        # Get the pairwise potentials
        J_pair_22_set = [np.linalg.inv(Q) for Q in Q_set]
        J_pair_21_set = [-J22.dot(A) for A,J22 in zip(A_set, J_pair_22_set)]
        J_pair_11_set = [A.T.dot(-J21) for A,J21 in zip(A_set, J_pair_21_set)]

        J_pair_11 = stack_set(J_pair_11_set)[self.stateseq]
        J_pair_21 = stack_set(J_pair_21_set)[self.stateseq]
        J_pair_22 = stack_set(J_pair_22_set)[self.stateseq]
        return J_pair_11, J_pair_21, J_pair_22

    @property
    def info_emission_params(self):

        expand = lambda a: a[None,...]
        stack_set = lambda x: np.concatenate(map(expand, x))


        # TODO: Double check this
        # TODO: Check for diagonal emissions
        C_set = [d.A for d in self.emission_distns]
        R_set = [d.sigma for d in self.emission_distns]
        RC_set = [np.linalg.solve(R, C) for C,R in zip(C_set, R_set)]
        CRC_set = [C.T.dot(RC) for C,RC in zip(C_set, RC_set)]

        J_node = stack_set(CRC_set)[self.stateseq]

        # TODO: Faster to replace this with a loop?
        RC = stack_set(RC_set)[self.stateseq]
        h_node = np.einsum('ni,nij->nj', self.data, RC)

        return J_node, h_node

    @property
    def info_params(self):
        return self.info_init_params + self.info_dynamics_params + self.info_emission_params

    @property
    def extra_info_params(self):
        J_init = np.linalg.inv(self.sigma_init)
        h_init = np.linalg.solve(self.sigma_init, self.mu_init)

        Q_set = [d.sigma for d in self.dynamics_distns]
        logdet_pairs = [-np.linalg.slogdet(Q)[1] for Q in Q_set]

        expand = lambda a: a[None, ...]
        stack_set = lambda x: np.concatenate(map(expand, x))
        logdet_pairs = stack_set(logdet_pairs)[self.stateseq]

        # Observations
        if self.diagonal_noise:
            # Use the fact that the diagonalregression prior is factorized
            rsq_set = [d.sigmasq_flat for d in self.emission_distns]
            rsq = stack_set(rsq_set)[self.stateseq]

            J_yy = 1./rsq
            logdet_node = -np.sum(np.log(rsq), axis=1)

        else:

            R_set = [d.sigma for d in self.emission_distns]
            R_inv_set = [np.linalg.inv(R) for R in R_set]
            R_logdet_set = [-np.linalg.slogdet(R)[1] for R in R_set]

            J_yy = stack_set(R_inv_set)[self.stateseq]
            logdet_node = stack_set(R_logdet_set)[self.stateseq]

        return J_init, h_init, logdet_pairs, J_yy, logdet_node, self.data

    def smooth(self):
        # Use the info E step because it can take advantage of diagonal noise
        # The standard E step could but we have not implemented it
        self.info_E_step()
        if self.model._single_emission:
            return self.smoothed_mus.dot(self.emission_distns[0].A.T)
        else:
            # TODO: Improve this
            return np.array([C.dot(mu) for C,mu in zip(self.Cs, self.smoothed_mus)])

    def info_E_step(self):
        self._gaussian_normalizer, self.smoothed_mus, \
        self.smoothed_sigmas, E_xtp1_xtT = \
            info_E_step(*self.info_params)

        self._gaussian_normalizer += LDSStates._info_extra_loglike_terms(
            *self.extra_info_params,
            isdiag=self.diagonal_noise)
        #
        # self._set_expected_stats(
        #     self.smoothed_mus, self.smoothed_sigmas, E_xtp1_xtT)


######################
#  algorithm mixins  #
######################

class _SLDSStatesGibbs(_SLDSStates):
    @property
    def aBl(self):
        if self._aBl is None:
            aBl = self._aBl = np.empty((self.T, self.num_states))
            ids, dds, eds = self.init_dynamics_distns, self.dynamics_distns, \
                self.emission_distns

            for idx, (d1, d2, d3) in enumerate(zip(ids, dds, eds)):
                # Initial state distribution
                aBl[0,idx] = d1.log_likelihood(self.gaussian_states[0])

                # Dynamics
                xs = np.hstack((self.gaussian_states[:-1], self.inputs[:-1]))
                aBl[:-1,idx] = d2.log_likelihood((xs, self.gaussian_states[1:]))

                # Emissions
                xs = np.hstack((self.gaussian_states, self.inputs))
                aBl[:,idx] += d3.log_likelihood((xs, self.data))

            aBl[np.isnan(aBl).any(1)] = 0.
        return self._aBl

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
            # filter_and_sample(
            #     self.mu_init, self.sigma_init,
            #     self.As, self.Bs, self.sigma_statess,
            #     self.Cs, self.Ds, self.sigma_obss,
            #     self.inputs, self.data)
        info_sample(*self.info_params)
        self._gaussian_normalizer += LDSStates._info_extra_loglike_terms(
            *self.extra_info_params, isdiag=self.diagonal_noise)

class _SLDSStatesMeanField(_SLDSStates):
    @property
    def expected_info_dynamics_params(self):
        def get_paramseq(distns):
            contract = partial(np.tensordot, self.expected_states, axes=1)
            params = [d.meanfield_expectedstats() for d in distns]
            return map(contract, zip(*params))

        J_pair_22, J_pair_21, J_pair_11, logdet_pair = \
            get_paramseq(self.dynamics_distns)

        return J_pair_11, J_pair_21, J_pair_22

    @property
    def expected_info_emission_params(self):
        # Now get the expected observation potentials
        def get_paramseq(distns):
            contract = partial(np.tensordot, self.expected_states, axes=1)
            params = [d.meanfield_expectedstats() for d in distns]
            return map(contract, zip(*params))

        J_yy, J_yx, J_node, logdet_node = get_paramseq(self.emission_distns)
        h_node = np.einsum('ni,nij->nj', self.data, J_yx)

        return J_node, h_node

    @property
    def expected_info_params(self):
        return self.info_init_params + \
               self.expected_info_dynamics_params + \
               self.expected_info_emission_params

    @property
    def expected_extra_info_params(self):
        J_init = np.linalg.inv(self.sigma_init)
        h_init = np.linalg.solve(self.sigma_init, self.mu_init)

        def get_paramseq(distns):
            contract = partial(np.tensordot, self.expected_states, axes=1)
            std_param = lambda d: d._natural_to_standard(d.mf_natural_hypparam)
            params = [mniw_expectedstats(*std_param(d)) for d in distns]
            return map(contract, zip(*params))

        expand = lambda a: a[None, ...]
        stack_set = lambda x: np.concatenate(map(expand, x))

        _, _, _, logdet_pairs = \
            get_paramseq(self.dynamics_distns)

        # Observations
        if self.diagonal_noise:
            E_sigmasq_inv_set = [d.mf_expectations[2] for d in self.emission_distns]
            J_yy = self.expected_states.dot(stack_set(E_sigmasq_inv_set))

            E_logdet_sigma_set = [-np.sum(d.mf_expectations[3]) for d in self.emission_distns]
            logdet_node = self.expected_states.dot(stack_set(E_logdet_sigma_set))

        else:
            J_yy, _, _, logdet_node = get_paramseq(self.emission_distns)

        return J_init, h_init, logdet_pairs, J_yy, logdet_node, self.data


    @property
    def mf_aBl(self):
        if self._mf_aBl is None:
            mf_aBl = self._mf_aBl = np.empty((self.T, self.num_states))
            ids, dds, eds = self.init_dynamics_distns, self.dynamics_distns, \
                self.emission_distns

            # TODO: Update
            for idx, (d1, d2, d3) in enumerate(zip(ids, dds, eds)):
                mf_aBl[0,idx] = d1.expected_log_likelihood(
                    stats=(self.smoothed_mus[0], self.ExxT[0], 1.))
                mf_aBl[:-1,idx] = d2.expected_log_likelihood(
                    stats=self.E_dynamics_stats)
                mf_aBl[:,idx] += d3.expected_log_likelihood(
                    stats=self.E_emission_stats)

            mf_aBl[np.isnan(mf_aBl).any(1)] = 0.
        return self._mf_aBl

    def meanfieldupdate(self, niter=1):
        niter = self.niter if hasattr(self, 'niter') else niter
        for itr in range(niter):
            self.meanfield_update_discrete_states()
            self.meanfield_update_gaussian_states()

    def _init_mf_from_gibbs(self):
        super(_SLDSStatesMeanField, self)._init_mf_from_gibbs()
        self.meanfield_update_gaussian_states()

    def meanfield_update_discrete_states(self):
        super(_SLDSStatesMeanField, self).meanfieldupdate()

    def meanfield_update_gaussian_states(self):
        # TODO: Handle inputs
        # J_init = np.linalg.inv(self.sigma_init)
        # h_init = np.linalg.solve(self.sigma_init, self.mu_init)
        #
        # def get_paramseq(distns):
        #     contract = partial(np.tensordot, self.expected_states, axes=1)
        #     std_param = lambda d: d._natural_to_standard(d.mf_natural_hypparam)
        #     params = [mniw_expectedstats(*std_param(d)) for d in distns]
        #     return map(contract, zip(*params))
        #
        # J_pair_22, J_pair_21, J_pair_11, logdet_pair = \
        #     get_paramseq(self.dynamics_distns)
        # J_yy, J_yx, J_node, logdet_node = get_paramseq(self.emission_distns)
        # h_node = np.einsum('ni,nij->nj', self.data, J_yx)
        self._mf_lds_normalizer, self.smoothed_mus, self.smoothed_sigmas, \
            E_xtp1_xtT = info_E_step(*self.expected_info_params)

        self._mf_lds_normalizer += LDSStates._info_extra_loglike_terms(
            *self.expected_extra_info_params, isdiag=self.diagonal_noise)

        self._set_gaussian_expected_stats(
            self.smoothed_mus,self.smoothed_sigmas,E_xtp1_xtT)

    def E_step(self):
        # TODO: Update normalizer?
        self._normalizer, self.smoothed_mus, self.smoothed_sigmas, \
        E_xtp1_xtT = E_step(
            self.mu_init, self.sigma_init,
            self.As, self.Bs, self.sigma_statess,
            self.Cs, self.Ds, self.sigma_obss,
            self.inputs, self.data)

        self._set_gaussian_expected_stats(
            self.smoothed_mus, self.smoothed_sigmas, E_xtp1_xtT)

    def _set_gaussian_expected_stats(self, smoothed_mus, smoothed_sigmas, E_xtp1_xtT):
        # TODO: Handle inputs

        assert not np.isnan(E_xtp1_xtT).any()
        assert not np.isnan(smoothed_mus).any()
        assert not np.isnan(smoothed_sigmas).any()

        # this is like LDSStates._set_expected_states but doesn't sum over time
        T = self.T
        ExxT = self.ExxT = smoothed_sigmas \
                           + self.smoothed_mus[:, :, None] * self.smoothed_mus[:, None, :]

        # Initial state stats
        self.E_init_stats = (self.smoothed_mus[0], ExxT[0], 1.)

        # Dynamics stats
        # TODO avoid memory instantiation by adding to Regression (2TD vs TD^2)
        # TODO only compute EyyT once
        E_xtp1_xtp1T = self.E_xtp1_xtp1T = ExxT[1:]
        E_xt_xtT = self.E_xt_xtT = ExxT[:-1]

        self.E_dynamics_stats = \
            (E_xtp1_xtp1T, E_xtp1_xtT, E_xt_xtT, np.ones(T-1))

        # Emission stats
        if self.diagonal_noise:
            EyyT = self.EyyT = self.data**2
            EyxT = self.EyxT = self.data[:, :, None] * self.smoothed_mus[:, None, :]
            self.E_emission_stats = (EyyT, EyxT, ExxT, np.ones(T))
        else:
            EyyT = self.EyyT = self.data[:, :, None] * self.data[:, None, :]
            EyxT = self.EyxT = self.data[:, :, None] * self.smoothed_mus[:, None, :]
            self.E_emission_stats = (EyyT, EyxT, ExxT, np.ones(T))

        self._mf_aBl = None  # TODO

    def get_vlb(self, most_recently_updated=False):
        if not most_recently_updated:
            raise NotImplementedError  # TODO
        else:
            # TODO hmm_vlb term is jumpy
            hmm_vlb = super(_SLDSStatesMeanField, self).get_vlb(
                most_recently_updated=False)
            return hmm_vlb + self._mf_lds_normalizer


class _SLDSStatesMaskedData(_SLDSStatesGibbs, _SLDSStatesMeanField):
    def __init__(self, model, T=None, data=None, mask=None, stateseq=None, gaussian_states=None,
                 generate=True, initialize_from_prior=True, fixed_stateseq=None):
        if mask is not None:
            assert mask.shape == data.shape
            self.mask = mask
        elif data is not None and np.any(np.isnan(data)):
            from warnings import warn
            warn("data includes NaN's. Treating these as missing data.")
            self.mask = ~np.isnan(data)
            # TODO: Remove this necessity
            warn("zeroing out nans in data to make sure code works")
            data[np.isnan(data)] = 0
        else:
            self.mask = None

        super(_SLDSStatesMaskedData, self).__init__(model, T=T, data=data, stateseq=stateseq,
                                                    gaussian_states=gaussian_states, generate=generate,
                                                    initialize_from_prior=initialize_from_prior,
                                                    fixed_stateseq=fixed_stateseq)

    @property
    def info_emission_params(self):
        if self.mask is None:
            return super(_SLDSStatesMaskedData, self).info_emission_params

        expand = lambda a: a[None,...]
        stack_set = lambda x: np.concatenate(map(expand, x))

        if self.diagonal_noise:
            sigmasq_set = [d.sigmasq_flat for d in self.emission_distns]
            sigmasq = stack_set(sigmasq_set)[self.stateseq]
            J_obs = self.mask / sigmasq

            C_set = [d.A for d in self.emission_distns]
            CCT_set = [np.array([np.outer(cp, cp) for cp in C]).
                           reshape((self.D_emission, self.D_latent**2))
                       for C in C_set]

            J_node = np.zeros((self.T, self.D_latent**2))
            h_node = np.zeros((self.T, self.D_latent))
            for i in range(len(self.emission_distns)):
                ti = np.where(self.stateseq == i)[0]
                J_node[ti] = np.dot(J_obs[ti], CCT_set[i])
                h_node[ti] = (self.data[ti] * J_obs[ti]).dot(C_set[i])

            J_node = J_node.reshape((self.T, self.D_latent, self.D_latent))

        else:
            raise NotImplementedError("Only supporting diagonal regression class right now")

        return J_node, h_node

    @property
    def extra_info_params(self):
        params = super(_SLDSStatesMaskedData, self).extra_info_params

        # Mask off missing data entries -- should work?
        if self.mask is not None:
            params = params[:-1] + (self.data * self.mask, )

        return params

    @property
    def expected_info_emission_params(self):
        if self.mask is None:
            return super(_SLDSStatesMaskedData, self).expected_info_emission_params

        if self.diagonal_noise:
            expand = lambda a: a[None, ...]
            stack_set = lambda x: np.concatenate(map(expand, x))

            # E_C, E_CCT, E_sigmasq_inv, _ = self.emission_distn.mf_expectations
            mf_params = [d.mf_expectations for d in self.emission_distns]

            E_C_set = [prms[0] for prms in mf_params]
            E_CCT_set = [np.reshape(prms[1], (self.D_emission, self.D_latent ** 2))
                       for prms in mf_params]
            E_sigmasq_inv_set = [prms[2] for prms in mf_params]

            E_sigmasq_inv = stack_set(E_sigmasq_inv_set)[self.stateseq]
            J_obs = self.mask * E_sigmasq_inv

            J_node = np.zeros((self.T, self.D_latent**2))
            h_node = np.zeros((self.T, self.D_latent))
            for i in range(len(self.emission_distns)):
                J_node += self.expected_states[:,i][:,None] * np.dot(J_obs, E_CCT_set[i])
                h_node += self.expected_states[:,i][:,None] * (self.data * J_obs).dot(E_C_set[i])

            J_node = J_node.reshape((self.T, self.D_latent, self.D_latent))

        else:
            raise NotImplementedError("Only supporting diagonal regression class right now")

        return J_node, h_node

    @property
    def expected_extra_info_params(self):
        params = super(_SLDSStatesMaskedData, self).expected_extra_info_params

        # Mask off missing data entries -- should work?
        if self.mask is not None:
            params = params[:-1] + (self.data * self.mask, )

        return params

    @property
    def aBl(self):
        if self._aBl is None:
            aBl = self._aBl = np.empty((self.T, self.num_states))
            ids, dds, eds = self.init_dynamics_distns, self.dynamics_distns, \
                self.emission_distns

            for idx, (d1, d2, d3) in enumerate(zip(ids, dds, eds)):
                aBl[0,idx] = d1.log_likelihood(self.gaussian_states[0])
                aBl[:-1,idx] = d2.log_likelihood(self.strided_gaussian_states)
                aBl[:,idx] += d3.log_likelihood((self.gaussian_states, self.data), mask=self.mask)

            aBl[np.isnan(aBl).any(1)] = 0.
        return self._aBl

    def _set_gaussian_expected_stats(self, smoothed_mus, smoothed_sigmas, E_xtp1_xtT):
        if self.mask is None:
            super(_SLDSStatesMaskedData, self).\
                _set_gaussian_expected_stats(smoothed_mus, smoothed_sigmas, E_xtp1_xtT)

        assert not np.isnan(E_xtp1_xtT).any()
        assert not np.isnan(smoothed_mus).any()
        assert not np.isnan(smoothed_sigmas).any()

        # Same as in parent class
        # this is like LDSStates._set_expected_states but doesn't sum over time
        T = self.T
        ExxT = self.ExxT = smoothed_sigmas \
                           + self.smoothed_mus[:, :, None] * self.smoothed_mus[:, None, :]

        # Initial state stats
        self.E_init_stats = (self.smoothed_mus[0], ExxT[0], 1.)

        # Dynamics stats
        # TODO avoid memory instantiation by adding to Regression (2TD vs TD^2)
        # TODO only compute EyyT once
        E_xtp1_xtp1T = self.E_xtp1_xtp1T = ExxT[1:]
        E_xt_xtT = self.E_xt_xtT = ExxT[:-1]

        self.E_dynamics_stats = \
            (E_xtp1_xtp1T, E_xtp1_xtT, E_xt_xtT, np.ones(T - 1))

        # Emission stats
        masked_data = self.data * self.mask if self.mask is not None else self.data
        if self.diagonal_noise:
            Eysq = self.EyyT = masked_data ** 2
            EyxT = self.EyxT = masked_data[:, :, None] * self.smoothed_mus[:, None, :]
            ExxT = self.mask[:,:,None,None] * ExxT[:,None,:,:]
            self.E_emission_stats = (Eysq, EyxT, ExxT, self.mask)
        else:
            raise Exception("Only DiagonalRegression currently supports missing data")

        self._mf_aBl = None  # TODO


####################
#  states classes  #
####################

class HMMSLDSStatesPython(
    _SLDSStatesMaskedData,
    _SLDSStatesGibbs,
    _SLDSStatesMeanField,
    HMMStatesPython):
    pass


class HMMSLDSStatesEigen(
    _SLDSStatesMaskedData,
    _SLDSStatesGibbs,
    _SLDSStatesMeanField,
    HMMStatesEigen):
    pass


class HSMMSLDSStatesPython(
    _SLDSStatesMaskedData,
    _SLDSStatesGibbs,
    _SLDSStatesMeanField,
    HSMMStatesPython):
    pass


class HSMMSLDSStatesEigen(
    _SLDSStatesMaskedData,
    _SLDSStatesGibbs,
    _SLDSStatesMeanField,
    HSMMStatesEigen):
    pass


class GeoHSMMSLDSStates(
    _SLDSStatesMaskedData,
    _SLDSStatesGibbs,
    _SLDSStatesMeanField,
    GeoHSMMStates):
    pass
