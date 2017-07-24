from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pybasicbayes.distributions import Regression, Gaussian
from pybasicbayes.util.text import progprint_xrange

from pypolyagamma.distributions import BernoulliRegression
from pyslds.models import HMMCountSLDS, WeakLimitStickyHDPHMMCountSLDS

npr.seed(0)
cmap = "jet"

### Hyperparameters
K, Kmax, D_obs, D_latent = 2, 10, 10, 2
mu_init = np.zeros(D_latent)
mu_init[0] = 1.0
sigma_init = 0.01 * np.eye(D_latent)

# Create an SLDS with stable dynamics matrices
def random_rotation(n,theta):
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    out = np.zeros((n,n))
    out[:2,:2] = rot
    q = np.linalg.qr(np.random.randn(n,n))[0]
    return q.dot(out).dot(q.T)

As = [random_rotation(D_latent, np.pi/24.),
     random_rotation(D_latent, np.pi/8.)]

# Start with a random emission matrix
C = np.random.randn(D_obs, D_latent)
b = -2.0 * np.ones((D_obs, 1))

init_dynamics_distns = [Gaussian(mu=mu_init, sigma=sigma_init) for _ in range(K)]
dynamics_distns = [Regression(A=A, sigma=0.01*np.eye(D_latent)) for A in As]
emission_distns = BernoulliRegression(D_obs, D_latent, A=C, b=b)

truemodel = HMMCountSLDS(
    dynamics_distns=dynamics_distns,
    emission_distns=emission_distns,
    init_dynamics_distns=init_dynamics_distns,
    alpha=3., init_state_distn='uniform')

### Generate data from an SLDS
# Manually create the states object with the mask
T = 1000
stateseq = np.repeat(np.arange(T//100) % 2, 100).astype(np.int32)
statesobj = truemodel._states_class(model=truemodel, T=stateseq.size, stateseq=stateseq)
statesobj.generate_gaussian_states()
data = statesobj.data = statesobj.generate_obs()

# Manually mask off chunks of data
mask = np.ones_like(data, dtype=bool)
chunksz = 50
for i,offset in enumerate(range(0,T,chunksz)):
    j = i % (D_obs + 1)
    if j < D_obs:
        mask[offset:min(offset+chunksz, T), j] = False
    if j == D_obs:
        mask[offset:min(offset+chunksz, T), :] = False
statesobj.mask = mask
truemodel.states_list.append(statesobj)

### Make a model
model = WeakLimitStickyHDPHMMCountSLDS(
    init_dynamics_distns=
        [Gaussian(
            nu_0=5, sigma_0=3.*np.eye(D_latent),
            mu_0=np.zeros(D_latent), kappa_0=0.01,
            mu=mu_init, sigma=sigma_init
        ) for _ in range(Kmax)],
    dynamics_distns=
        [Regression(
            A=np.eye(D_latent), sigma=np.eye(D_latent),
            nu_0=D_latent+3,
            S_0=D_latent*np.eye(D_latent),
            M_0=np.zeros((D_latent, D_latent)),
            K_0=D_latent*np.eye(D_latent),
        ) for _ in range(Kmax)],
    emission_distns=BernoulliRegression(D_obs, D_latent),
    alpha=3., gamma=3.0, kappa=100., init_state_distn='uniform')
model.add_data(data=data, mask=mask)

### Run a Gibbs sampler
N_samples = 500
def gibbs_update(model):
    model.resample_model()
    smoothed_obs = model.states_list[0].smooth()
    return model.log_likelihood(), model.stateseqs[0], smoothed_obs

lls, z_smpls, smoothed_obss = \
    zip(*[gibbs_update(model) for _ in progprint_xrange(N_samples)])

### Plot the log likelihood over iterations
plt.figure(figsize=(10,6))
plt.plot(lls,'-b')
plt.plot([0,N_samples], truemodel.log_likelihood() * np.ones(2), '-k')
plt.xlabel('iteration')
plt.ylabel('log likelihood')

### Plot the smoothed observations
fig = plt.figure(figsize=(10,10))
N_subplots = min(D_obs,6)
gs = gridspec.GridSpec(N_subplots*2+1,1)

zax = fig.add_subplot(gs[0])
zax.imshow(truemodel.stateseqs[0][None,:], aspect='auto', cmap=cmap)
zax.set_ylabel("Discrete \nstate", labelpad=20, multialignment="center", rotation=90)
zax.set_xticklabels([])
zax.set_yticks([])

given_data = data.copy()
given_data[~mask] = np.nan
masked_data = data.copy()
masked_data[mask] = np.nan
ylims = (-1.1*abs(data).max(), 1.1*abs(data).max())
xlims = (0, min(T,1000))

n_to_plot = np.arange(min(5, D_obs))
for i,j in enumerate(n_to_plot):
    ax = fig.add_subplot(gs[1+2*i:1+2*(i+1)])
    # Plot spike counts
    given_ts = np.where(given_data[:,j]==1)[0]
    ax.plot(given_ts, np.ones_like(given_ts), 'ko', markersize=5)

    masked_ts = np.where(masked_data[:,j]==1)[0]
    ax.plot(masked_ts, np.ones_like(masked_ts), 'o', markerfacecolor="gray", markeredgecolor="none", markersize=5)

    # Plot the inferred rate
    ax.plot([0], [0], 'b', lw=2, label="smoothed obs.")
    ax.plot(smoothed_obss[-1][:,j], 'r', lw=2, label="smoothed pr.")

    # Overlay the mask
    ax.imshow(1-mask[:,j][None,:],cmap="Greys",alpha=0.25,extent=(0,T) + ylims, aspect="auto")

    if i == 0:
        plt.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 2.))
    if i == N_subplots - 1:
        plt.xlabel('time index')
    ax.set_xlim(xlims)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("$x_%d(t)$" % (j+1))

### Plot the discrete state samples
fig = plt.figure(figsize=(8,4))
gs = gridspec.GridSpec(6,1)
ax1 = fig.add_subplot(gs[:-1])
ax2 = fig.add_subplot(gs[-1])

im = ax1.imshow(np.array(z_smpls), aspect='auto', interpolation="none", cmap=cmap)
ax1.autoscale(False)
ax1.set_ylabel("Iteration")
ax1.set_xticks([])

ax2.imshow(truemodel.stateseqs[0][None,:], aspect='auto', cmap=cmap)
ax2.set_ylabel("True", labelpad=27)
ax2.set_xlabel("Time")
ax2.set_yticks([])

fig.suptitle("Discrete state samples")

plt.show()

