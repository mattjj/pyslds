from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

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
      for alpha, theta in ((0.95,0.2), (0.95,-0.2), (1., 0.))]

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

Nmax = 5           # number of latnt discrete states
P = 2              # latent linear dynamics' dimension
D = data.shape[1]  # data dimension

dynamics_distns = [
    AutoRegression(
        A=np.eye(P),sigma=np.eye(P),
        nu_0=2,S_0=2.*np.eye(P),M_0=np.eye(P),K_0=10.*np.eye(P))
    for _ in xrange(Nmax)]

emission_distns = [
    Regression(
        A=np.eye(D),sigma=0.05*np.eye(D),
        nu_0=20.,S_0=20.*np.eye(D),M_0=np.eye(2),K_0=0.01*np.eye(P))
    for _ in xrange(Nmax)]


init_dynamics_distns = [
    Gaussian(nu_0=3,sigma_0=3.*np.eye(P),mu_0=np.zeros(P),kappa_0=0.01)
    for _ in xrange(Nmax)]

model = WeakLimitStickyHDPHMMSLDS(
    dynamics_distns=dynamics_distns,
    emission_distns=emission_distns,
    init_dynamics_distns=init_dynamics_distns,
    kappa=5.,alpha=10.,gamma=20.,init_state_distn='uniform')


##################
#  run sampling  #
##################

model.add_data(data)

# cheating!
# model.add_data(data, stateseq=labels)
# for _ in progprint_xrange(1000):
#     model.states_list[0].resample_gaussian_states()
#     model.resample_parameters()


def resample(itr):
    # model.resample_model()
    model.states_list[0].resample_gaussian_states()
    model.states_list[0].resample_discrete_states()
    return model.stateseqs[0]


def resample2(itr):
    model.resample_model()
    return model.stateseqs[0]

# TODO show truth in a separate imshow with same axis
# samples[200:] = labels

n_show = 100
samples = np.empty((n_show, data.shape[0]))
samples[:n_show] = model.stateseqs[0]

im = plt.matshow(samples[::-1])
fig = plt.gcf()
ax = plt.gca()

ax.autoscale(False)

plt.draw()
plt.ion()
plt.show()

from matplotlib.transforms import Bbox

xo, yo, w, ht = ax.bbox.bounds
h = ht / samples.shape[0]

from itertools import count
for itr in count():
    if itr >= 0:
        model.resample_model()
    else:
        model.resample_states()
        model.resample_hmm_parameters()

        # don't resample emission distns yet!
        model.resample_dynamics_distns()
        model.resample_init_dynamics_distns()
    samples[itr % n_show] = model.stateseqs[0]

    im.set_array(samples[::-1])
    ax.draw_artist(im)
    fig.canvas.blit(Bbox.from_bounds(xo,yo+h*(itr % n_show),w,h+1))
