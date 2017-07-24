from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from pyhsmm.basic.distributions import PoissonDuration

from autoregressive.models import ARHSMM
from autoregressive.distributions import AutoRegression
from pyhsmm.util.text import progprint_xrange

from pybasicbayes.distributions import DiagonalRegression, Gaussian, Regression

from pyslds.models import HMMSLDS

np.random.seed(0)


###################
#  generate data  #
###################
T = 1000
Kmax = 10       # number of latent discrete states
D_latent = 2    # latent linear dynamics' dimension
D_input = 1     # latent linear dynamics' dimension
D_obs = 2       # data dimension
N_iter = 200    # number of VBEM iterations

As = [np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
      for alpha, theta in ((0.95,0.1), (0.95,-0.1), (1., 0.))]

truemodel = ARHSMM(
    alpha=4., init_state_concentration=4.,
    obs_distns=[AutoRegression(A=A, sigma=0.05*np.eye(2)) for A in As],
    dur_distns=[PoissonDuration(alpha_0=3*50, beta_0=3) for _ in As])

truemodel.prefix = np.array([[0., 3.]])
data, labels = truemodel.generate(T)
data = data[truemodel.nlags:]

plt.figure()
plt.plot(data[:,0], data[:,1], 'x-')


#################
#  build model  #
#################
Cs = [np.eye(D_obs) for _ in range(Kmax)]                   # Shared emission matrices
sigma_obss = [0.05 * np.eye(D_obs) for _ in range(Kmax)]    # Emission noise covariances

model = HMMSLDS(
    init_dynamics_distns=
        [Gaussian(
            nu_0=5, sigma_0=3.*np.eye(D_latent),
            mu_0=np.zeros(D_latent), kappa_0=0.01,
            mu=np.zeros(D_latent), sigma=np.eye(D_latent)
        ) for _ in range(Kmax)],
    dynamics_distns=
        [Regression(
            A=np.hstack((np.eye(D_latent), np.zeros((D_latent, D_input)))),
            sigma=np.eye(D_latent),
            nu_0=D_latent+3,
            S_0=D_latent*np.eye(D_latent),
            M_0=np.hstack((np.eye(D_latent), np.zeros((D_latent, D_input)))),
            K_0=D_latent*np.eye(D_latent + D_input),
        ) for _ in range(Kmax)],
    emission_distns=
        DiagonalRegression(
            D_obs, D_latent + D_input,
            alpha_0=2.0, beta_0=1.0,
        ),
    alpha=3., init_state_distn='uniform')

model.add_data(data, inputs=np.ones((T, D_input)))
model.resample_states()

for _ in progprint_xrange(0):
    model.resample_model()
model.states_list[0]._init_mf_from_gibbs()


####################
#  run mean field  #
####################

vlbs = []
for _ in progprint_xrange(N_iter):
    model.VBEM_step()
    vlbs.append(model.VBEM_ELBO())
    if len(vlbs) > 1:
        assert vlbs[-1] > vlbs[-2] - 1e-8

plt.figure()
plt.plot(vlbs)
# plt.plot([0, N_iter], truemodel.log_likelihood(data) * np.ones(2), '--k')
plt.xlabel("Iteration")
plt.ylabel("VLB")

import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(9,3))
gs = gridspec.GridSpec(7,1)
ax1 = fig.add_subplot(gs[:-2])
ax2 = fig.add_subplot(gs[-2], sharex=ax1)
ax3 = fig.add_subplot(gs[-1], sharex=ax1)

im = ax1.matshow(model.states_list[0].expected_states.T, aspect='auto', cmap="Greys")
ax1.set_xticks([])
ax1.set_yticks(np.arange(Kmax))
ax1.set_ylabel("Discrete State")

ax2.imshow(model.states_list[0].expected_states.argmax(1)[None,:],
           vmin=0, vmax=Kmax, aspect='auto')
ax2.set_xticks([])
ax2.set_yticks([])

ax3.matshow(labels[None,:], aspect='auto')
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_xlabel("Time")


plt.show()
