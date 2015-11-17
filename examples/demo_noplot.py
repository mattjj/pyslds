from __future__ import division
import numpy as np

from pyhsmm.basic.distributions import Regression, Gaussian, PoissonDuration
from autoregressive.distributions import AutoRegression
from pyhsmm.util.text import progprint_xrange

from pyslds.models import WeakLimitStickyHDPHMMSLDS

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

data_list = []
label_list = []
for _ in xrange(4):
    data, labels = truemodel.generate(400)
    data = data[truemodel.nlags:]
    data_list.append(data)
    label_list.append(labels)


#################
#  build model  #
#################

Nmax = 10          # number of latnt discrete states
P = 2              # latent linear dynamics' dimension
D = data.shape[1]  # data dimension

dynamics_distns = [
    AutoRegression(
        A=np.eye(P),sigma=np.eye(P),
        nu_0=10., S_0=10.*np.eye(P),M_0=np.eye(P),K_0=10.*np.eye(P))
    for _ in xrange(Nmax)]

emission_distns = [
    Regression(
        A=np.eye(D),sigma=0.05*np.eye(D),
        nu_0=8.,S_0=np.eye(P),M_0=np.eye(P),K_0=10.*np.eye(P))
    for _ in xrange(Nmax)]


init_dynamics_distns = [
    Gaussian(nu_0=8.,sigma_0=3.*np.eye(P),mu_0=np.zeros(P),kappa_0=0.01)
    for _ in xrange(Nmax)]

model = WeakLimitStickyHDPHMMSLDS(
    dynamics_distns=dynamics_distns,
    emission_distns=emission_distns,
    init_dynamics_distns=init_dynamics_distns,
    kappa=100.,alpha=3.,gamma=3.,init_state_distn='uniform')

for data in data_list:
    model.add_data(data)
model.resample_states()

##################
#  run sampling  #
##################

from matplotlib.transforms import Bbox
import matplotlib.gridspec as gridspec

n_show = 50
samples = np.empty((n_show, data.shape[0]))
samples[:n_show] = model.stateseqs[0]

from itertools import count
for itr in xrange(10):
    print "model ll: %2.2f"%model.log_likelihood()
    model.resample_model(num_procs=2)

