from __future__ import division
import numpy as np
from functools import partial
from builtins import zip

from pybasicbayes.distributions.regression import DiagonalRegression

import pyhsmm
from pyhsmm.util.general import list_split

from pyslds.states import HMMSLDSStatesPython, HMMSLDSStatesEigen, HSMMSLDSStatesPython, \
    HSMMSLDSStatesEigen


from pybasicbayes.abstractions import Distribution

class _SLDSMixin(object):
    def __init__(self,dynamics_distns,emission_distns,init_dynamics_distns,**kwargs):
        self.init_dynamics_distns = init_dynamics_distns
        self.dynamics_distns = dynamics_distns

        # Allow for a single, shared emission distribution
        if isinstance(emission_distns, Distribution):
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

    def _generate_obs(self,s):
        if s.data is None:
            s.data = s.generate_obs()
        else:
            # TODO: Handle missing data
            raise NotImplementedError

        return s.data, s.gaussian_states

    @property
    def diagonal_noise(self):
        return all([isinstance(ed, DiagonalRegression) for ed in self.emission_distns])

    @property
    def has_missing_data(self):
        return any([s.mask is not None for s in self.states_list])

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
            mask = [s.mask for s in self.states_list] \
                if self.has_missing_data else None

            self._emission_distn.resample(
                data=[(s.gaussian_states, s.inputs, s.data)
                      for s in self.states_list],
                mask=mask)
        else:
            for state, d in enumerate(self.emission_distns):
                mask = [s.mask[s.stateseq == state] for s in self.states_list] \
                    if self.has_missing_data else None

                d.resample(
                    data=[(s.gaussian_states[s.stateseq == state],
                           s.inputs[s.stateseq == state],
                           s.data[s.stateseq == state])
                          for s in self.states_list],
                    mask=mask)
        self._clear_caches()

    def resample_obs_distns(self):
        pass  # handled in resample_parameters

    ### joblib parallel

    def _joblib_resample_states(self,states_list,num_procs):
        # TODO: Update to handle inputs
        from joblib import Parallel, delayed
        import parallel

        if len(states_list) > 0:
            joblib_args = map(self._get_joblib_pair, states_list)

            parallel.model = self
            parallel.args = list_split(joblib_args, num_procs)

            idxs = range(len(parallel.args))
            raw_stateseqs = Parallel(n_jobs=num_procs,backend='multiprocessing')\
                    (map(delayed(parallel._get_sampled_stateseq), idxs))

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
        sum_tuples = lambda lst: map(sum, zip(*lst))
        E_stats = lambda i, s: \
            tuple(s.expected_states[0,i] * stat for stat in s.E_init_stats)

        for state, d in enumerate(self.init_dynamics_distns):
            d.meanfieldupdate(
                stats=sum_tuples(E_stats(state, s) for s in self.states_list))


    def meanfield_update_dynamics_distns(self):
        contract = partial(np.tensordot, axes=1)
        sum_tuples = lambda lst: map(sum, zip(*lst))
        E_stats = lambda i, s: \
            tuple(contract(s.expected_states[1:,i], stat) for stat in s.E_dynamics_stats)

        for state, d in enumerate(self.dynamics_distns):
            d.meanfieldupdate(
                stats=sum_tuples(E_stats(state, s) for s in self.states_list))

    def meanfield_update_emission_distns(self):
        sum_tuples = lambda lst: map(sum, zip(*lst))

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
