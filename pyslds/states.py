from __future__ import division
import numpy as np

from pyhsmm.internals.hmm_states import HMMStatesPython, HMMStatesEigen
from pyhsmm.internals.hsmm_states import HSMMStatesPython, HSMMStatesEigen, \
    GeoHSMMStates

from autoregressive.util import AR_striding
from pylds.lds_messages_interface import filter_and_sample


###########
#  bases  #
###########

class _SLDSStates(object):
    ### generation

    def generate_states(self):
        super(_SLDSStates,self).generate_states()
        self.generate_gaussian_states()

    def generate_gaussian_states(self):
        # TODO this is dumb, but generating from the prior will be unstable
        self.gaussian_states = np.random.normal(size=(self.T,self.D_latent))

    def generate_obs(self):
        raise NotImplementedError

    ## convenience properties

    @property
    def strided_gaussian_states(self):
        return AR_striding(self.gaussian_states,1)

    @property
    def D_latent(self):
        return self.dynamics_distns[0].D

    @property
    def D_emission(self):
        return self.emission_distns[0].D

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
        for itr in xrange(niter):
            self.resample_discrete_states()
            self.resample_gaussian_states()

    def resample_discrete_states(self):
        super(_SLDSStatesGibbs, self).resample()

    def resample_gaussian_states(self):
        self._aBl = None  # clear any caching
        self._normalizer, self.gaussian_states = filter_and_sample(
            self.mu_init, self.sigma_init,
            self.As, self.BBTs, self.Cs, self.DDTs,
            self.data)


class _SLDSStatesMeanField(_SLDSStates):
    @property
    def mf_aBl(self):
        # TODO check that expected_log_likelihood calls in Gaussian and
        # Regression work like this
        if self._mf_aBl is None:
            mf_aBl = self._mf_aBl = np.empty((self.T, self.num_states))
            ids, eds, dds = self.init_dynamics_distns, self.emission_distns, \
                self.dynamics_distns
            Ees, Eds = self.E_emission_stats, self.E_dynamics_stats

            for idx, (d1, d2) in enumerate(zip(ids, eds)):
                mf_aBl[0,idx] = d1.expected_log_likelihood(
                    stats=(self.smoothed_mus[0],self.smoothed_sigmas[0]))
                mf_aBl[0,idx] += d2.expected_log_likelihood(
                    stats=tuple(stat[0] for stat in Ees))

            for idx, (d1, d2) in enumerate(zip(dds, eds)):
                mf_aBl[1:,idx] = d1.expected_log_likelihood(stats=Eds)
                mf_aBl[1:,idx] += d2.expected_log_likelihood(
                    stats=tuple(stat[1:] for stat in Ees))

            mf_aBl[np.isnan(mf_aBl).any(1)] = 0.
        return self._mf_aBl

    def meanfieldupdate(self, niter=1):
        # TODO make niter an __init__ arg instead of a method arg here
        for itr in xrange(niter):
            self.meanfield_update_discrete_states()
            self.meanfield_update_gaussian_states()

    def meanfield_update_discrete_states(self):
        super(_SLDSStatesMeanField, self).meanfieldupdate()

    def meanfield_update_gaussian_states(self):
        # TODO like meanfieldupdate in pylds.states.LDSStates
        raise NotImplementedError

    def vlb(self):
        raise NotImplementedError  # TODO

    def _set_expected_stats(self, smoothed_mus, smoothed_sigmas, E_xtp1_xtT):
        assert not np.isnan(E_xtp1_xtT).any()
        assert not np.isnan(smoothed_mus).any()
        assert not np.isnan(smoothed_sigmas).any()

        # TODO avoid memory instantiation by adding to Regression (2TD vs TD^2)
        EyyT = self.data[:,:,None] * self.data[:,None,:]  # TODO compute once
        EyxT = self.data[:,:,None] * self.smoothed_mus[:,None,:]
        ExxT = smoothed_sigmas \
            + self.smoothed_mus[:,:,None] * self.smoothed_mus[:,None,:]

        E_xtp1_xtp1T = ExxT[1:]
        E_xt_xtT = ExxT[:-1]

        T = self.T
        self.E_emission_stats = (EyyT, EyxT, ExxT, np.ones(T))
        self.E_dynamics_stats = \
            (E_xtp1_xtp1T, E_xtp1_xtT, E_xt_xtT, np.ones(T-1))


####################
#  states classes  #
####################

class HMMSLDSStatesPython(_SLDSStatesGibbs, HMMStatesPython):
    pass


class HMMSLDSStatesEigen(_SLDSStatesGibbs, HMMStatesEigen):
    pass


class HSMMSLDSStatesPython(_SLDSStatesGibbs, HSMMStatesPython):
    pass


class HSMMSLDSStatesEigen(_SLDSStatesGibbs, HSMMStatesEigen):
    pass


class GeoHSMMSLDSStates(_SLDSStatesGibbs, GeoHSMMStates):
    pass
