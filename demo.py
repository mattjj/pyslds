from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

import pyhsmm
from pyhsmm.util.text import progprint_xrange
from pyhsmm.basic.distributions import Regression, Gaussian
from autoregressive.distributions import AutoRegression

from slds.models import HMMSLDS

###################
#  generate data  #
###################

import autoregressive

As = [np.array(
    [[np.cos(theta), -np.sin(theta)],
     [np.sin(theta), np.cos(theta)]]
    ) for theta in (0.05,-0.05,0.15)] + [np.eye(2)]

truemodel = autoregressive.models.ARHSMM(
        alpha=4.,init_state_concentration=4.,
        obs_distns=[AutoRegression(A=A,sigma=np.eye(2)) for A in As],
        dur_distns=[pyhsmm.basic.distributions.PoissonDuration(alpha_0=4*25,beta_0=4)
            for _ in As],
        )

data, labels = truemodel.generate(300)
data = data[1:]

plt.figure()
plt.plot(data[:,0],data[:,1],'bx-')

#################
#  build model  #
#################

Nmax = len(As)

dynamics_distns = [
        AutoRegression(
            A=np.eye(2),sigma=2*np.eye(2),
            nu_0=5,S_0=np.eye(2),M_0=np.zeros((2,2)),K_0=np.eye(2),
            )
        for _ in xrange(Nmax)]
emission_distns = [
        Regression(
            A=np.eye(2),sigma=np.eye(2),
            nu_0=10,S_0=np.eye(2),M_0=np.zeros((2,2)),K_0=np.eye(2)
            )
        for _ in xrange(Nmax)]
init_dynamics_distns = [
        Gaussian(nu_0=5,sigma_0=np.eye(2),mu_0=np.zeros(2),kappa_0=1.)
        for _ in xrange(Nmax)]

model = HMMSLDS(
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns,
        init_dynamics_distns=init_dynamics_distns,
        alpha=4.,init_state_concentration=1.)

##################
#  run sampling  #
##################

np.seterr(over='raise')

model.add_data(data,stateseq=labels) # TODO needs init when passing in labels
s = model.states_list[0]
s.resample_gaussian_states()

for itr in progprint_xrange(50):
    model.resample_model()

plt.plot(s.gaussian_states[:,0],s.gaussian_states[:,1],'r-')

plt.show()
