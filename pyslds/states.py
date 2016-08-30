from __future__ import division
import numpy as np
from functools import partial

from pybasicbayes.util.stats import mniw_expectedstats

from pyhsmm.internals.hmm_states import HMMStatesPython, HMMStatesEigen, _StatesBase
from pyhsmm.internals.hsmm_states import HSMMStatesPython, HSMMStatesEigen, \
    GeoHSMMStates

from autoregressive.util import AR_striding
from pylds.states import LDSStates
from pylds.lds_messages_interface import filter_and_sample, info_E_step

# TODO on instantiating, maybe gaussian states should be resampled
# TODO make niter an __init__ arg instead of a method arg


###########
#  bases  #
###########

class _SLDSStates(object):
    def __init__(self,model,T=None,data=None,stateseq=None,gaussian_states=None,
            generate=True,initialize_from_prior=True,fixed_stateseq=None):
        self.model = model

        self.T = T if T is not None else data.shape[0]
        self.data = data
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
        # Generate from the prior and raise exception if unstable
        T, n = self.T, self.D_latent

        # The discrete stateseq should be populated by the super call above
        dss = self.stateseq

        gss = np.empty((T,n),dtype='double')
        gss[0] = self.init_dynamics_distns[dss[0]].rvs()

        for t in range(1,T):
            gss[t] = self.dynamics_distns[dss[t]].\
                rvs(lagged_data=gss[t-1][None,:])
            assert np.all(np.isfinite(gss[t])), "SLDS appears to be unstable!"

        self.gaussian_states = gss

    def generate_obs(self):
        # Go through each time bin, get the discrete latent state,
        # use that to index into the emission_distns to get samples
        T, p = self.T, self.D_emission
        dss, gss = self.stateseq, self.gaussian_states
        data = np.empty((T,p),dtype='double')

        for t in range(self.T):
            data[t] = self.emission_distns[dss[t]].\
                rvs(x=gss[t][None,:], return_xy=False)

        return data

    ## convenience properties

    @property
    def strided_gaussian_states(self):
        return AR_striding(self.gaussian_states,1)

    @property
    def D_latent(self):
        return self.dynamics_distns[0].D

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
    def mu_init(self):
        return self.init_dynamics_distns[self.stateseq[0]].mu

    @property
    def sigma_init(self):
        return self.init_dynamics_distns[self.stateseq[0]].sigma

    @property
    def As(self):
        Aset = np.concatenate([d.A[None,...] for d in self.dynamics_distns])
        return Aset[self.stateseq]

    @property
    def BBTs(self):
        Bset = np.concatenate([d.sigma[None,...] for d in self.dynamics_distns])
        return Bset[self.stateseq]

    @property
    def Cs(self):
        Cset = np.concatenate([d.A[None,...] for d in self.emission_distns])
        return Cset[self.stateseq]

    @property
    def DDTs(self):
        Dset = np.concatenate([d.sigma[None,...] for d in self.emission_distns])
        return Dset[self.stateseq]

    @property
    def _kwargs(self):
        return dict(super(_SLDSStates, self)._kwargs,
                    stateseq=self.stateseq,
                    gaussian_states=self.gaussian_states)

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
                aBl[0,idx] = d1.log_likelihood(self.gaussian_states[0])
                aBl[:-1,idx] = d2.log_likelihood(self.strided_gaussian_states)
                aBl[:,idx] += d3.log_likelihood((self.gaussian_states, self.data))

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
            filter_and_sample(
                self.mu_init, self.sigma_init,
                self.As, self.BBTs, self.Cs, self.DDTs,
                self.data)


class _SLDSStatesMeanField(_SLDSStates):
    @property
    def mf_aBl(self):
        if self._mf_aBl is None:
            mf_aBl = self._mf_aBl = np.empty((self.T, self.num_states))
            ids, dds, eds = self.init_dynamics_distns, self.dynamics_distns, \
                self.emission_distns

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
        J_init = np.linalg.inv(self.sigma_init)
        h_init = np.linalg.solve(self.sigma_init, self.mu_init)

        def get_paramseq(distns):
            contract = partial(np.tensordot, self.expected_states, axes=1)
            std_param = lambda d: d._natural_to_standard(d.mf_natural_hypparam)
            params = [mniw_expectedstats(*std_param(d)) for d in distns]
            return map(contract, zip(*params))

        J_pair_22, J_pair_21, J_pair_11, logdet_pair = \
            get_paramseq(self.dynamics_distns)
        J_yy, J_yx, J_node, logdet_node = get_paramseq(self.emission_distns)
        h_node = np.einsum('ni,nij->nj', self.data, J_yx)

        self._mf_lds_normalizer, self.smoothed_mus, self.smoothed_sigmas, \
            E_xtp1_xtT = info_E_step(
                J_init,h_init,J_pair_11,-J_pair_21,J_pair_22,J_node,h_node)
        self._mf_lds_normalizer += LDSStates._info_extra_loglike_terms(
            J_init, h_init, logdet_pair, J_yy, logdet_node, self.data)

        self._set_gaussian_expected_stats(
            self.smoothed_mus,self.smoothed_sigmas,E_xtp1_xtT)

    def _set_gaussian_expected_stats(self, smoothed_mus, smoothed_sigmas, E_xtp1_xtT):
        assert not np.isnan(E_xtp1_xtT).any()
        assert not np.isnan(smoothed_mus).any()
        assert not np.isnan(smoothed_sigmas).any()

        # this is like LDSStates._set_expected_states but doesn't sum over time

        # TODO avoid memory instantiation by adding to Regression (2TD vs TD^2)
        # TODO only compute EyyT once
        EyyT = self.EyyT = self.data[:,:,None] * self.data[:,None,:]
        EyxT = self.EyxT = self.data[:,:,None] * self.smoothed_mus[:,None,:]
        ExxT = self.ExxT = smoothed_sigmas \
            + self.smoothed_mus[:,:,None] * self.smoothed_mus[:,None,:]

        E_xtp1_xtp1T = self.E_xtp1_xtp1T = ExxT[1:]
        E_xt_xtT = self.E_xt_xtT = ExxT[:-1]

        T = self.T
        self.E_emission_stats = (EyyT, EyxT, ExxT, np.ones(T))
        self.E_dynamics_stats = \
            (E_xtp1_xtp1T, E_xtp1_xtT, E_xt_xtT, np.ones(T-1))
        self.E_init_stats = (self.smoothed_mus[0], ExxT[0], 1.)

        self._mf_aBl = None  # TODO

    def get_vlb(self, most_recently_updated=False):
        if not most_recently_updated:
            raise NotImplementedError  # TODO
        else:
            # TODO hmm_vlb term is jumpy
            hmm_vlb = super(_SLDSStatesMeanField, self).get_vlb(
                most_recently_updated=False)
            return hmm_vlb + self._mf_lds_normalizer


####################
#  states classes  #
####################

class HMMSLDSStatesPython(
        _SLDSStatesGibbs,
        _SLDSStatesMeanField,
        HMMStatesPython):
    pass


class HMMSLDSStatesEigen(
        _SLDSStatesGibbs,
        _SLDSStatesMeanField,
        HMMStatesEigen):
    pass


class HSMMSLDSStatesPython(
        _SLDSStatesGibbs,
        _SLDSStatesMeanField,
        HSMMStatesPython):
    pass


class HSMMSLDSStatesEigen(
        _SLDSStatesGibbs,
        _SLDSStatesMeanField,
        HSMMStatesEigen):
    pass


class GeoHSMMSLDSStates(
        _SLDSStatesGibbs,
        _SLDSStatesMeanField,
        GeoHSMMStates):
    pass
