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


As = [np.hstack((-np.eye(2),2*np.eye(2))),
        np.array([[np.cos(np.pi/6),-np.sin(np.pi/6)],[np.sin(np.pi/6),np.cos(np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2))),
        np.array([[np.cos(-np.pi/6),-np.sin(-np.pi/6)],[np.sin(-np.pi/6),np.cos(-np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2)))]

import autoregressive
truemodel = autoregressive.models.ARHSMM(
        alpha=4.,init_state_concentration=4.,
        obs_distns=[AutoRegression(A=A,sigma=0.1*np.eye(2)) for A in As],
        dur_distns=[pyhsmm.basic.distributions.PoissonDuration(alpha_0=4*25,beta_0=4)
            for state in range(len(As))],
        )

data, labels = truemodel.generate(100)

plt.figure()
plt.plot(data[:,0],data[:,1],'bx-')

#################
#  build model  #
#################

Nmax = 4

dynamics_distns = [
        AutoRegression(nu_0=3,S_0=np.eye(2),M_0=np.zeros((2,2)),K_0=np.eye(2))
        for _ in xrange(Nmax)]
emission_distns = [
        Regression(nu_0=3,S_0=np.eye(2),M_0=np.zeros((2,2)),K_0=np.eye(2))
        for _ in xrange(Nmax)]
init_dynamics_distns = [
        Gaussian(nu_0=3,sigma_0=np.eye(2),mu_0=np.zeros(2),kappa_0=1.)
        for _ in xrange(Nmax)]

model = HMMSLDS(
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns,
        init_dynamics_distns=init_dynamics_distns,
        alpha=4.,init_state_concentration=1.)

##################
#  run sampling  #
##################

model.add_data(data)
s = model.states_list[0]
s.gaussian_states = data

for itr in progprint_xrange(50):
    model.resample_dynamics_distns()
    s.resample_discrete_states()

