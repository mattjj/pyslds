from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pybasicbayes.distributions import Regression, DiagonalRegression, Gaussian
from pybasicbayes.util.text import progprint_xrange

from pyslds.models import HMMSLDS, WeakLimitStickyHDPHMMSLDS

npr.seed(0)

#########################
#  set some parameters  #
#########################
K, Kmax, D_obs, D_latent = 2, 10, 4, 2
mu_init = np.zeros(D_latent)
mu_init[0] = 1.0
sigma_init = 0.01*np.eye(D_latent)

def random_rotation(n,theta):
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    out = np.zeros((n,n))
    out[:2,:2] = rot
    q = np.linalg.qr(np.random.randn(n,n))[0]
    return q.dot(out).dot(q.T)


As = [0.99 * random_rotation(D_latent, np.pi/24.),
      0.99 * random_rotation(D_latent, np.pi/12.)]

C = np.random.randn(D_obs, D_latent)
sigma_obs = 0.5 * np.ones(D_obs)


###################
#  generate data  #
###################
init_dynamics_distns = [Gaussian(mu=mu_init, sigma=sigma_init) for _ in range(K)]
dynamics_distns = [Regression(A=A, sigma=np.eye(D_latent)) for A in As]
emission_distns = DiagonalRegression(D_obs, D_latent, A=C, sigmasq=sigma_obs)

truemodel = HMMSLDS(
    dynamics_distns=dynamics_distns,
    emission_distns=emission_distns,
    init_dynamics_distns=init_dynamics_distns,
    alpha=3., init_state_distn='uniform')

# Manually create the states object with the mask
T = 1000
stateseq = np.repeat(np.arange(T//100) % 2, 100).astype(np.int32)
statesobj = truemodel._states_class(model=truemodel, T=stateseq.size, stateseq=stateseq)
statesobj.generate_gaussian_states()
data = statesobj.data = statesobj.generate_obs()
gaussian_states = statesobj.gaussian_states
truemodel.states_list.append(statesobj)

# Mask off a chunk of data
# mask = npr.rand(*data.shape) < 0.5
mask = np.ones_like(data, dtype=bool)
chunksz = 200
for i,offset in enumerate(range(0,T,chunksz)):
    j = i % (D_obs + 1)
    if j < D_obs:
        mask[offset:min(offset+chunksz, T), j] = False
    if j == D_obs:
        mask[offset:min(offset+chunksz, T), :] = False
statesobj.mask = mask

###############
#  make model #
###############
model = HMMSLDS(
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
            M_0=np.eye(D_latent),
            K_0=D_latent*np.eye(D_latent),
        ) for _ in range(Kmax)],
    emission_distns=
        DiagonalRegression(
            D_obs, D_latent,
            alpha_0=2.0, beta_0=1.0,
        ),
    alpha=3., init_state_distn='uniform')
model.add_data(data=data, mask=mask)

###############
#  fit model  #
###############
N_init_samples = 0
for _ in progprint_xrange(N_init_samples):
    model.resample_model()
model._init_mf_from_gibbs()

N_iters = 100
def update(model):
    model.VBEM_step()
    # model.meanfield_coordinate_descent_step()
    lp = model.log_likelihood()
    smoothed_obs = model.states_list[0].smooth()
    return lp, model.stateseqs[0], smoothed_obs

# Fit the model
lls, z_smpls, smoothed_obss = zip(*[update(model) for _ in progprint_xrange(N_iters)])

################
# likelihoods  #
################
plt.figure(figsize=(10,6))
plt.plot(lls[1:],'-b')
plt.plot([0, N_iters - 1], truemodel.log_likelihood() * np.ones(2), '-k')
plt.xlabel('iteration')
plt.ylabel('log likelihood')

################
#  smoothing   #
################
plt.figure(figsize=(10,6))
given_data = data.copy()
given_data[~mask] = np.nan
masked_data = data.copy()
masked_data[mask] = np.nan
ylims = (-1.1*abs(data).max(), 1.1*abs(data).max())
xlims = (0, min(T,1000))

N_subplots = min(D_obs,4)
for i in range(N_subplots):
    plt.subplot(N_subplots,1,i+1,aspect="auto")

    plt.plot(given_data[:,i], 'k', label="observed")
    plt.plot(masked_data[:,i], ':k', label="masked")
    plt.plot(smoothed_obss[-1][:,i], 'b', lw=2, label="smoothed")

    plt.imshow(1-mask[:,i][None,:],cmap="Greys",alpha=0.25,extent=(0,T) + ylims, aspect="auto")

    if i == 0:
        plt.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.5))

    if i == N_subplots - 1:
        plt.xlabel('time index')

    plt.ylabel("$x_%d(t)$" % (i+1))
    plt.ylim(ylims)
    plt.xlim(xlims)
# plt.savefig("slds_missing_data_ex.png")

################
#  z samples   #
################
fig = plt.figure(figsize=(8,4))
gs = gridspec.GridSpec(6,1)
ax1 = fig.add_subplot(gs[:-1])
ax2 = fig.add_subplot(gs[-1])

im = ax1.imshow(np.array(z_smpls), aspect='auto', interpolation="none")
ax1.autoscale(False)
ax1.set_ylabel("Iteration")
ax1.set_xticks([])

ax2.imshow(truemodel.stateseqs[0][None,:], aspect='auto')
ax2.set_ylabel("True", labelpad=27)
ax2.set_xlabel("Time")
ax2.set_yticks([])

fig.suptitle("Discrete state samples")
# plt.savefig("slds_discrete_states.png")

plt.show()

