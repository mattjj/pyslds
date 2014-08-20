from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import pyhsmm
from pyhsmm.basic.distributions import Regression
from autoregressive.distributions import AutoRegression, Gaussian

class _SLDSMixin(object):
    def __init__(self,dynamics_distns,emission_distns,init_dynamics_distns,**kwargs):
        self.init_dynamics_distns = init_dynamics_distns
        self.dynamics_distns = dynamics_distns
        self.emission_distns = emission_distns
        super(_SLDSMixin,self).__init__(obs_distns=self.dynamics_distns,**kwargs)

    def resample_parameters(self):
        self.resample_emission_distns()
        self.resample_init_dynamics_distns()
        super(_SLDSMixin,self).resample_parameters()

    def resample_init_dynamics_distns():
        raise NotImplementedError

    def resample_emission_distns(self):
        raise NotImplementedError

