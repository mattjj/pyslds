from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from pyhsmm.basic.distributions import Regression, Gaussian, PoissonDuration
from autoregressive.distributions import AutoRegression
from pyhsmm.util.text import progprint_xrange

from pyslds.models import HMMSLDS

np.random.seed(0)


###################
#  generate data  #
###################

import autoregressive

As = [np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
      for alpha, theta in ((0.95,0.1), (0.95,-0.1), (1., 0.))]

truemodel = autoregressive.models.ARHSMM(
    alpha=4.,init_state_concentration=4.,
    obs_distns=[AutoRegression(A=A,sigma=0.05*np.eye(2)) for A in As],
    dur_distns=[PoissonDuration(alpha_0=5*50,beta_0=5) for _ in As])

truemodel.prefix = np.array([[0.,3.]])
data, labels = truemodel.generate(1000)
data = data[truemodel.nlags:]

plt.figure()
plt.plot(data[:,0],data[:,1],'bx-')


#################
#  build model  #
#################

Nmax = 10
P = 2
D = data.shape[1]

dynamics_distns = [
    AutoRegression(
        A=np.eye(P),sigma=np.eye(P),
        nu_0=3,S_0=3.*np.eye(P),M_0=np.eye(P),K_0=10.*np.eye(P))
    for _ in xrange(Nmax)]

emission_distns = [
    Regression(
        A=np.eye(D),sigma=0.05*np.eye(D),
        nu_0=5.,S_0=np.eye(P),M_0=np.eye(P),K_0=10.*np.eye(P))
    for _ in xrange(Nmax)]


init_dynamics_distns = [
    Gaussian(nu_0=4,sigma_0=4.*np.eye(P),mu_0=np.zeros(P),kappa_0=0.1)
    for _ in xrange(Nmax)]

model = HMMSLDS(
    dynamics_distns=dynamics_distns,
    emission_distns=emission_distns,
    init_dynamics_distns=init_dynamics_distns,
    alpha=3.,init_state_distn='uniform')

model.add_data(data)
model.resample_states()
for _ in progprint_xrange(5):
    model.resample_model()
model.states_list[0]._init_mf_from_gibbs()


####################
#  run mean field  #
####################

plt.figure()
plt.plot([model.meanfield_coordinate_descent_step()
          for _ in progprint_xrange(50)])

plt.figure()
plt.imshow(model.states_list[0].expected_states.T)

plt.show()
