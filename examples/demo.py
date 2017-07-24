from __future__ import division
import numpy as np
np.random.seed(0)

import matplotlib
# matplotlib.use("macosx")  # might be necessary for animation to work
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import autoregressive
from pyhsmm.basic.distributions import PoissonDuration
from pybasicbayes.distributions import AutoRegression

from pyslds.models import DefaultSLDS


###################
#  generate data  #
###################
As = [np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
      for alpha, theta in ((0.95,0.1), (0.95,-0.1), (1., 0.))]

truemodel = autoregressive.models.ARHSMM(
    alpha=4., init_state_concentration=4.,
    obs_distns=[AutoRegression(A=A, sigma=0.05*np.eye(2)) for A in As],
    dur_distns=[PoissonDuration(alpha_0=5*50, beta_0=5) for _ in As])

truemodel.prefix = np.array([[0.,3.]])
data, labels = truemodel.generate(1000)
data = data[truemodel.nlags:]

plt.figure()
plt.plot(data[:,0],data[:,1],'x-')
plt.xlabel("$y_1$")
plt.ylabel("$y_2$")


#################
#  build model  #
#################
Kmax = 10                           # number of latent discrete states
D_latent = 2                        # latent linear dynamics' dimension
D_obs = 2                           # data dimension

Cs = np.eye(D_obs)                  # Shared emission matrix
sigma_obss = 0.05 * np.eye(D_obs)   # Emission noise covariance

model = DefaultSLDS(
    K=Kmax, D_obs=D_obs, D_latent=D_latent,
    Cs=Cs, sigma_obss=sigma_obss)

model.add_data(data)
model.resample_states()


##################
#  run sampling  #
##################
n_show = 50
samples = np.empty((n_show, data.shape[0]))
samples[:n_show] = model.stateseqs[0]

fig = plt.figure(figsize=(8,3))
gs = gridspec.GridSpec(6,1)
ax1 = fig.add_subplot(gs[:-1])
ax2 = fig.add_subplot(gs[-1], sharex=ax1)

im = ax1.matshow(samples[::-1], aspect='auto')
ax1.autoscale(False)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_ylabel("Discrete State")
xo, yo, w, ht = ax1.bbox.bounds
h = ht / n_show

ax2.matshow(labels[None,:], aspect='auto')
ax2.set_xticks([])
ax2.set_xlabel("Time")
ax2.set_yticks([])

plt.draw()
plt.ion()
plt.show()


print("Press Ctrl-C to stop...")
from itertools import count
for itr in count():
    model.resample_model()

    samples[itr % n_show] = model.stateseqs[0]
    im.set_array(samples[::-1])
    plt.pause(0.001)
