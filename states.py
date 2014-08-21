from __future__ import division
import numpy as np

from pyhsmm.internals.hmm_states import HMMStatesPython, HMMStatesEigen
from pyhsmm.internals.hsmm_states import HSMMStatesPython, HSMMStatesEigen, \
        GeoHSMMStates, _SeparateTransMixin

from autoregressive.util import AR_striding

class _SLDSStatesMixin(object):
    ### convenience properties

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

    ### main stuff

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
            aBl = self._aBl = np.empty((self.gaussian_states.shape[0],self.num_states))

            for idx, distn in enumerate(self.init_dynamics_distns):
                aBl[0,idx] = distn.log_likelihood(self.gaussian_states[0])

            for idx, distn in enumerate(self.dynamics_distns):
                aBl[1:,idx] = distn.log_likelihood(self.strided_gaussian_states)

            aBl[np.isnan(aBl).any(1)] = 0.
        return self._aBl

    ## resampling conditionally linear dynamics

    def resample_gaussian_states(self):
        self._aBl = None # need to clear any caching
        init_mu, init_sigma = \
                self.dynamics_distns[self.stateseq[0]].mu, \
                self.dynamics_distns[self.stateseq[0]].sigma
        As, BBTs, Cs, DDTs = map(np.array,zip(*[(
            self.dynamics_distns[state].A,
            self.dynamics_distns[state].sigma,
            self.emission_distns[state].A,
            self.emission_distns[state].sigma,
            ) for state in self.stateseq[1:]]))
        self.gaussian_states = kf_resample_slds(
                init_mu=init_mu, init_sigma=init_sigma,
                As=As, BBTs=BBTs, Cs=Cs, DDTs=DDTs,
                emissions=self.data)

    @property
    def strided_gaussian_states(self):
        return AR_striding(self.gaussian_states,1)

    ## generation

    def generate_states(self):
        super(_SLDSStatesMixin,self).generate_states()
        self.generate_gaussian_states()

    def generate_gaussian_states(self):
        raise NotImplementedError

    def generate_obs(self):
        raise NotImplementedError

class HMMSLDSStates(_SLDSStatesMixin,HMMStatesPython):
    pass

class HMMSLDSStatesEigen(_SLDSStatesMixin,HMMStatesEigen):
    pass

class HSMMSLDSStates(_SLDSStatesMixin,HSMMStatesPython):
    pass

class HSMMSLDSStatesEigen(_SLDSStatesMixin,HSMMStatesEigen):
    pass

class GeoHSMMSLDSStates(_SLDSStatesMixin,GeoHSMMStates):
    pass


class HSMMSLDSStatesPossibleChangepointsSeparateTrans(
        _SLDSStatesMixin,_SeparateTransMixin,HSMMStatesPossibleChangepoints):
    pass


### kalman filtering and smoothing functions

def kf_resample_lds(init_mu,init_sigma,As,BBTs,Cs,DDTs,emissions):
    T, D_obs, D_latent = emissions.shape[0], emissions.shape[1], As[0].shape[0]

    filtered_mus = np.empty((T,D_latent))
    filtered_sigmas = np.empty((T,D_latent,D_latent))

    x = np.empty((T,D_latent))

    # messages forwards
    prediction_mu, prediction_sigma = init_mu, init_sigma
    for t, (A,BBT,C,DDT) in enumerate(zip(As,BBTs,Cs,DDTs)):
        # condition
        filtered_mus[t], filtered_sigmas[t] = \
            condition_on(prediction_mu,prediction_sigma,C,DDT,emissions[t])

        # predict
        prediction_mu, prediction_sigma = \
            A.dot(filtered_mus[t]), A.dot(filtered_sigmas[t]).dot(A.T) + BBT

    # sample backwards
    x[-1] = np.random.multivariate_normal(filtered_mus[-1],filtered_sigmas[-1])
    for t in xrange(T-2,-1,-1):
        x[t] = np.random.multivariate_normal(
                *condition_on(filtered_mus[t],filtered_sigmas[t],As[t],BBTs[t],x[t+1]))

    return x

def condition_on(mu_x,sigma_x,A,sigma_obs,y):
    sigma_xy = sigma_x.dot(A.T)
    sigma_yy = A.dot(sigma_x).dot(A.T) + sigma_obs
    mu = mu_x + sigma_xy.dot(solve_psd(sigma_yy, y - A.dot(mu_x)))
    sigma = sigma_x - sigma_xy.dot(solve_psd(sigma_yy,sigma_xy.T))
    return mu, sigma

solve_psd = np.linalg.solve

# TODO special code for diagonal plus low rank
# TODO test if psd solves are better

