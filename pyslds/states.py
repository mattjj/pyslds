from __future__ import division
import numpy as np

from pyhsmm.internals.hmm_states import HMMStatesPython, HMMStatesEigen
from pyhsmm.internals.hsmm_states import HSMMStatesPython, HSMMStatesEigen, \
    GeoHSMMStates

from autoregressive.util import AR_striding
from pylds.lds_messages_interface import filter_and_sample


class _SLDSStatesMixin(object):
    def resample(self,niter=1):
        for itr in xrange(niter):
            self.resample_discrete_states()
            self.resample_gaussian_states()

    ## resampling discrete states

    def resample_discrete_states(self):
        super(_SLDSStatesMixin,self).resample()

    @property
    def aBl(self):
        if self._aBl is None:
            aBl = self._aBl = np.empty(
                (self.gaussian_states.shape[0],self.num_states))

            for idx, distn in enumerate(self.init_dynamics_distns):
                aBl[0,idx] = distn.log_likelihood(self.gaussian_states[0])

            for idx, distn in enumerate(self.dynamics_distns):
                aBl[1:,idx] = distn.log_likelihood(
                    self.strided_gaussian_states)

            aBl[np.isnan(aBl).any(1)] = 0.
        return self._aBl

    ## resampling conditionally Gaussian dynamics

    def resample_gaussian_states(self):
        self._aBl = None  # clear any caching
        self._normalizer, self.gaussian_states = filter_and_sample(
            self.mu_init, self.sigma_init,
            self.As, self.BBTs, self.Cs, self.DDTs,
            self.data)

    @property
    def strided_gaussian_states(self):
        return AR_striding(self.gaussian_states,1)

    ## generation

    def generate_states(self):
        super(_SLDSStatesMixin,self).generate_states()
        self.generate_gaussian_states()

    def generate_gaussian_states(self):
        # TODO this is dumb, but generating from the prior will be unstable
        self.gaussian_states = np.random.normal(size=(self.T,self.D_latent))

    def generate_obs(self):
        raise NotImplementedError

    ## convenience properties

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


class HMMSLDSStatesPython(_SLDSStatesMixin,HMMStatesPython):
    pass


class HMMSLDSStatesEigen(_SLDSStatesMixin,HMMStatesEigen):
    pass


class HSMMSLDSStatesPython(_SLDSStatesMixin,HSMMStatesPython):
    pass


class HSMMSLDSStatesEigen(_SLDSStatesMixin,HSMMStatesEigen):
    pass


class GeoHSMMSLDSStates(_SLDSStatesMixin,GeoHSMMStates):
    pass
