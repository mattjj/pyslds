from __future__ import division
import numpy as np
from functools import partial
from builtins import zip

from pybasicbayes.distributions import DiagonalRegression, Gaussian, Regression

import pyhsmm
from pyhsmm.util.general import list_split

from pylds.util import random_rotation

from pyslds.states import HMMSLDSStatesPython, HMMSLDSStatesEigen, HSMMSLDSStatesPython, \
    HSMMSLDSStatesEigen


from pybasicbayes.abstractions import Distribution

class _SLDSMixin(object):
    def __init__(self,dynamics_distns,emission_distns,init_dynamics_distns,**kwargs):
        self.init_dynamics_distns = init_dynamics_distns
        self.dynamics_distns = dynamics_distns

        # Allow for a single, shared emission distribution
        if not isinstance(emission_distns, list):
            self._single_emission = True
            self._emission_distn = emission_distns
            self.emission_distns = [emission_distns] * len(self.dynamics_distns)
        else:
            assert isinstance(emission_distns, list) and \
                   len(emission_distns) == len(dynamics_distns)
            self._single_emission = False
            self.emission_distns = emission_distns

        super(_SLDSMixin,self).__init__(
            obs_distns=self.dynamics_distns,**kwargs)

    def generate(self, T=100, keep=True, **kwargs):
        s = self._states_class(model=self, T=T, initialize_from_prior=True, **kwargs)
        s.generate_states()
        data = self._generate_obs(s)
        if keep:
            self.states_list.append(s)
        return data + (s.stateseq,)


    def _generate_obs(self,s):
        if s.data is None:
            s.data = s.generate_obs()
        else:
            # TODO: Handle missing data
            raise NotImplementedError

        return s.data, s.gaussian_states

    def smooth(self, data, inputs=None, mask=None):
        self.add_data(data, inputs=inputs, mask=mask)
        s = self.states_list.pop()
        return s.smooth()

    @property
    def diagonal_noise(self):
        return all([isinstance(ed, DiagonalRegression) for ed in self.emission_distns])

    @property
    def has_missing_data(self):
        return any([s.mask is not None for s in self.states_list])

    @property
    def has_count_data(self):
        return any([hasattr(s, "omega") for s in self.states_list])

class _SLDSGibbsMixin(_SLDSMixin):
    def resample_parameters(self):
        self.resample_lds_parameters()
        self.resample_hmm_parameters()

    def resample_lds_parameters(self):
        self.resample_init_dynamics_distns()
        self.resample_dynamics_distns()
        self.resample_emission_distns()

    def resample_hmm_parameters(self):
        super(_SLDSGibbsMixin,self).resample_parameters()

    def resample_init_dynamics_distns(self):
        for state, d in enumerate(self.init_dynamics_distns):
            d.resample(
                [s.gaussian_states[0] for s in self.states_list
                    if s.stateseq[0] == state])
        self._clear_caches()

    def resample_dynamics_distns(self):
        zs = [s.stateseq[:-1] for s in self.states_list]
        xs = [np.hstack((s.gaussian_states[:-1], s.inputs[:-1]))
              for s in self.states_list]
        ys = [s.gaussian_states[1:] for s in self.states_list]

        for state, d in enumerate(self.dynamics_distns):
            d.resample(
                [(x[z == state], y[z == state])
                 for x, y, z in zip(xs, ys, zs)])
        self._clear_caches()

    def resample_emission_distns(self):
        if self._single_emission:
            data = [(np.hstack((s.gaussian_states, s.inputs)), s.data)
                    for s in self.states_list]
            mask = [s.mask for s in self.states_list] if self.has_missing_data else None
            omega = [s.omega for s in self.states_list] if self.has_count_data else None

            if self.has_count_data:
                self._emission_distn.resample(data=data, mask=mask, omega=omega)
            elif self.has_missing_data:
                self._emission_distn.resample(data=data, mask=mask)
            else:
                self._emission_distn.resample(data=data)
        else:
            for state, d in enumerate(self.emission_distns):
                data = [(np.hstack((s.gaussian_states[s.stateseq == state],
                                    s.inputs[s.stateseq == state])),
                         s.data[s.stateseq == state])
                        for s in self.states_list]

                mask = [s.mask[s.stateseq == state] for s in self.states_list] \
                    if self.has_missing_data else None
                omega = [s.omega[s.stateseq == state] for s in self.states_list] \
                    if self.has_count_data else None

                if self.has_count_data:
                    d.resample(data=data, mask=mask, omega=omega)
                elif self.has_missing_data:
                    d.resample(data=data, mask=mask)
                else:
                    d.resample(data=data)


        self._clear_caches()

    def resample_obs_distns(self):
        pass  # handled in resample_parameters

    ### joblib parallel

    def _joblib_resample_states(self,states_list,num_procs):
        # TODO: Update to handle inputs
        from joblib import Parallel, delayed
        import parallel

        if len(states_list) > 0:
            joblib_args = list(map(self._get_joblib_pair, states_list))

            parallel.model = self
            parallel.args = list_split(joblib_args, num_procs)

            idxs = range(len(parallel.args))
            raw_stateseqs = Parallel(n_jobs=num_procs,backend='multiprocessing')\
                    (list(map(delayed(parallel._get_sampled_stateseq), idxs)))

            flatten = lambda lst: [x for y in lst for x in y]
            raw_stateseqs = flatten(raw_stateseqs)

            # since list_split might reorder things, do the same to states_list
            states_list = flatten(list_split(states_list, num_procs))

            for s, tup in zip(states_list, raw_stateseqs):
                s.stateseq, s.gaussian_states, s._normalizer = tup


class _SLDSMeanFieldMixin(_SLDSMixin):
    def meanfield_update_parameters(self):
        self.meanfield_update_init_dynamics_distns()
        self.meanfield_update_dynamics_distns()
        self.meanfield_update_emission_distns()
        super(_SLDSMeanFieldMixin, self).meanfield_update_parameters()


    def meanfield_update_init_dynamics_distns(self):
        sum_tuples = lambda lst: list(map(sum, zip(*lst)))
        E_stats = lambda i, s: \
            tuple(s.expected_states[0,i] * stat for stat in s.E_init_stats)

        for state, d in enumerate(self.init_dynamics_distns):
            d.meanfieldupdate(
                stats=sum_tuples(E_stats(state, s) for s in self.states_list))


    def meanfield_update_dynamics_distns(self):
        contract = partial(np.tensordot, axes=1)
        sum_tuples = lambda lst: list(map(sum, zip(*lst)))
        E_stats = lambda i, s: \
            tuple(contract(s.expected_states[1:,i], stat) for stat in s.E_dynamics_stats)

        for state, d in enumerate(self.dynamics_distns):
            d.meanfieldupdate(
                stats=sum_tuples(E_stats(state, s) for s in self.states_list))

    def meanfield_update_emission_distns(self):
        sum_tuples = lambda lst: list(map(sum, zip(*lst)))

        if self._single_emission:
            E_stats = lambda s: \
                tuple(np.sum(stat, axis=0) for stat in s.E_emission_stats)

            self._emission_distn.meanfieldupdate(
                stats=sum_tuples(E_stats(s) for s in self.states_list))
        else:
            contract = partial(np.tensordot, axes=1)
            E_stats = lambda i, s: \
                tuple(contract(s.expected_states[:, i], stat) for stat in s.E_emission_stats)

            for state, d in enumerate(self.emission_distns):
                d.meanfieldupdate(
                    stats=sum_tuples(E_stats(state, s) for s in self.states_list))

    def meanfield_update_obs_distns(self):
        pass  # handled in meanfield_update_parameters

    ### init
    def _init_mf_from_gibbs(self):
        # Now also update the emission and dynamics params
        for ed in self.emission_distns:
            if hasattr(ed, "_initialize_mean_field"):
                ed._initialize_mean_field()
        for dd in self.dynamics_distns:
            if hasattr(dd, "_initialize_mean_field"):
                dd._initialize_mean_field()

        for s in self.states_list:
            s._init_mf_from_gibbs()

    ### vlb

    def vlb(self, states_last_updated=False):
        vlb = 0.
        vlb += sum(s.get_vlb(states_last_updated) for s in self.states_list)
        vlb += self.trans_distn.get_vlb()
        vlb += self.init_state_distn.get_vlb()
        vlb += sum(d.get_vlb() for d in self.init_dynamics_distns)
        vlb += sum(d.get_vlb() for d in self.dynamics_distns)
        if self._single_emission:
            vlb += self._emission_distn.get_vlb()
        else:
            vlb += sum(d.get_vlb() for d in self.emission_distns)
        return vlb


class HMMSLDSPython(_SLDSGibbsMixin, pyhsmm.models.HMMPython):
    _states_class = HMMSLDSStatesPython


class HMMSLDS(_SLDSGibbsMixin, _SLDSMeanFieldMixin, pyhsmm.models.HMM):
    _states_class = HMMSLDSStatesEigen


class HSMMSLDSPython(_SLDSGibbsMixin, pyhsmm.models.HSMMPython):
    _states_class = HSMMSLDSStatesPython


class HSMMSLDS(_SLDSGibbsMixin, pyhsmm.models.HSMM):
    _states_class = HSMMSLDSStatesEigen


class WeakLimitHDPHMMSLDS(_SLDSGibbsMixin, pyhsmm.models.WeakLimitHDPHMM):
    _states_class = HMMSLDSStatesEigen


class WeakLimitStickyHDPHMMSLDS(
        _SLDSGibbsMixin,
        _SLDSMeanFieldMixin,
        pyhsmm.models.WeakLimitStickyHDPHMM):
    _states_class = HMMSLDSStatesEigen


class WeakLimitHDPHSMMSLDS(_SLDSGibbsMixin, pyhsmm.models.WeakLimitHDPHSMM):
    _states_class = HSMMSLDSStatesEigen


## Default constructors

def _default_model(model_class, K, D_obs, D_latent, D_input=0,
                   mu_inits=None, sigma_inits=None,
                   As=None, Bs=None, sigma_statess=None,
                   Cs=None, Ds=None, sigma_obss=None,
                   alpha=3.0, init_state_distn='uniform',
                   **kwargs):

    # Initialize init_dynamics_distns
    init_dynamics_distns = \
        [Gaussian(nu_0=D_latent+3,
                  sigma_0=3.*np.eye(D_latent),
                  mu_0=np.zeros(D_latent),
                  kappa_0=0.01)
         for _ in range(K)]

    if mu_inits is not None:
        assert isinstance(mu_inits, list) and len(mu_inits) == K
        for id, mu in zip(init_dynamics_distns, mu_inits):
            id.mu = mu

    if sigma_inits is not None:
        assert isinstance(sigma_inits, list) and len(sigma_inits) == K
        for id, sigma in zip(init_dynamics_distns, sigma_inits):
            id.sigma = sigma

    # Initialize dynamics distributions
    dynamics_distns = [Regression(
        nu_0=D_latent + 1,
        S_0=D_latent * np.eye(D_latent),
        M_0=np.zeros((D_latent, D_latent + D_input)),
        K_0=D_latent * np.eye(D_latent + D_input))
        for _ in range(K)]
    if As is not None:
        assert isinstance(As, list) and len(As) == K
        if D_input > 0:
            assert isinstance(Bs, list) and len(Bs) == K
            As = [np.hstack((A, B)) for A,B in zip(As, Bs)]
    else:
        # As = [random_rotation(D_latent) for _ in range(K)]
        As = [np.eye(D_latent) for _ in range(K)]
        if D_input > 0:
            As = [np.hstack((A, np.zeros((D_latent, D_input))))
                  for A in As]
    for dd, A in zip(dynamics_distns, As):
        dd.A = A

    if sigma_statess is not None:
        assert isinstance(sigma_statess, list) and len(sigma_statess) == K
    else:
        sigma_statess = [np.eye(D_latent) for _ in range(K)]

    for dd, sigma in zip(dynamics_distns, sigma_statess):
        dd.sigma = sigma

    # Initialize emission distributions
    _single_emission = (Cs is not None) and (not isinstance(Cs, list))

    if _single_emission:
        if D_input > 0:
            assert Ds is not None and not isinstance(Ds, list)
            Cs = np.hstack((Cs, Ds))

        if sigma_obss is None:
            sigma_obss = np.eye(D_obs)

        emission_distns = Regression(
            nu_0=D_obs + 3,
            S_0=D_obs * np.eye(D_obs),
            M_0=np.zeros((D_obs, D_latent + D_input)),
            K_0=D_obs * np.eye(D_latent + D_input),
            A=Cs, sigma=sigma_obss)

    else:
        emission_distns = [Regression(
            nu_0=D_obs + 1,
            S_0=D_obs * np.eye(D_obs),
            M_0=np.zeros((D_obs, D_latent + D_input)),
            K_0=D_obs * np.eye(D_latent + D_input))
            for _ in range(K)]

        if Cs is not None and sigma_obss is not None:
            assert isinstance(Cs, list) and len(Cs) == K
            assert isinstance(sigma_obss, list) and len(sigma_obss) == K
            if D_input > 0:
                assert isinstance(Ds, list) and len(Ds) == K
                Cs = [np.hstack((C, D)) for C,D in zip(Cs, Ds)]
        else:
            Cs = [np.zeros((D_obs, D_latent + D_input)) for _ in range(K)]
            sigma_obss = [0.05 * np.eye(D_obs) for _ in range(K)]

        for ed, C, sigma in zip(emission_distns, Cs, sigma_obss):
            ed.A = C
            ed.sigma = sigma

    model = model_class(
        init_dynamics_distns=init_dynamics_distns,
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns,
        init_state_distn=init_state_distn,
        alpha=alpha,
        **kwargs)

    return model

def DefaultSLDS(K, D_obs, D_latent, D_input=0,
                mu_inits=None, sigma_inits=None,
                As=None, Bs=None, sigma_statess=None,
                Cs=None, Ds=None, sigma_obss=None,
                alpha=3.,
                **kwargs):
    return _default_model(HMMSLDS, K, D_obs, D_latent, D_input=D_input,
                          mu_inits=mu_inits, sigma_inits=sigma_inits,
                          As=As, Bs=Bs, sigma_statess=sigma_statess,
                          Cs=Cs, Ds=Ds, sigma_obss=sigma_obss,
                          alpha=alpha,
                          **kwargs)


def DefaultWeakLimitHDPSLDS(K, D_obs, D_latent, D_input=0,
                mu_inits=None, sigma_inits=None,
                As=None, Bs=None, sigma_statess=None,
                Cs=None, Ds=None, sigma_obss=None,
                alpha=3., gamma=3.,
                **kwargs):
    return _default_model(WeakLimitHDPHMMSLDS, K, D_obs, D_latent, D_input=D_input,
                          mu_inits=mu_inits, sigma_inits=sigma_inits,
                          As=As, Bs=Bs, sigma_statess=sigma_statess,
                          Cs=Cs, Ds=Ds, sigma_obss=sigma_obss,
                          alpha=alpha, gamma=gamma,
                          **kwargs)

def DefaultWeakLimitStickyHDPSLDS(K, D_obs, D_latent, D_input=0,
                mu_inits=None, sigma_inits=None,
                As=None, Bs=None, sigma_statess=None,
                Cs=None, Ds=None, sigma_obss=None,
                alpha=3., gamma=3., kappa=10.,
                **kwargs):
    return _default_model(WeakLimitStickyHDPHMMSLDS, K, D_obs, D_latent, D_input=D_input,
                          mu_inits=mu_inits, sigma_inits=sigma_inits,
                          As=As, Bs=Bs, sigma_statess=sigma_statess,
                          Cs=Cs, Ds=Ds, sigma_obss=sigma_obss,
                          kappa=kappa, alpha=alpha, gamma=gamma,
                          **kwargs)