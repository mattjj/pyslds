from __future__ import division
import numpy as np
np.seterr(divide="raise")
import matplotlib.pyplot as plt

from pyhsmm.basic.distributions import PoissonDuration
from autoregressive.distributions import AutoRegression
from pyhsmm.util.text import progprint_xrange

from pyslds.models import DefaultSLDS

# np.random.seed(0)


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
    dur_distns=[PoissonDuration(alpha_0=3*50,beta_0=3) for _ in As])

truemodel.prefix = np.array([[0.,3.]])
data, labels = truemodel.generate(1000)
data = data[truemodel.nlags:]

plt.figure()
plt.plot(data[:,0],data[:,1],'bx-')


#################
#  build model  #
#################

Kmax = 10                           # number of latent discrete states
D_latent = 2                        # latent linear dynamics' dimension
D_obs = 2                           # data dimension

Cs = [np.eye(D_obs) for _ in range(Kmax)]                   # Shared emission matrices
sigma_obss = [0.05 * np.eye(D_obs) for _ in range(Kmax)]    # Emission noise covariances

model = DefaultSLDS(
    K=Kmax, D_obs=D_obs, D_latent=D_latent,
    Cs=Cs, sigma_obss=sigma_obss)

model.add_data(data)
model.resample_states()

for _ in progprint_xrange(0):
    model.resample_model()
model.states_list[0]._init_mf_from_gibbs()


####################
#  run mean field  #
####################

vlbs = []
for _ in progprint_xrange(100):
    model.VBEM_step()
    vlbs.append(model.VBEM_ELBO())
    if len(vlbs) > 1:
        assert vlbs[-1] > vlbs[-2] - 1e-8

plt.figure()
plt.plot(vlbs)
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
