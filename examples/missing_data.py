from __future__ import division
import copy
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from pybasicbayes.distributions import AutoRegression, DiagonalRegression, Gaussian
from pybasicbayes.util.text import progprint_xrange

from pyslds.models import HMMSLDS

npr.seed(0)

from pybasicbayes.util.profiling import show_line_stats

#########################
#  set some parameters  #
#########################
K, D_obs, D_latent = 2, 4, 2
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


As = [random_rotation(D_latent, np.pi/24.),
     random_rotation(D_latent, np.pi/12.)]

C = np.random.randn(D_obs, D_latent)
sigma_obs = 0.1 * np.eye(D_obs)


###################
#  generate data  #
###################
init_dynamics_distns = [Gaussian(mu=mu_init, sigma=sigma_init) for _ in xrange(K)]
dynamics_distns = [AutoRegression(A=A, sigma=0.01*np.eye(D_latent)) for A in As]
emission_distns = DiagonalRegression(D_obs, D_latent, A=C, sigmasq=0.5*np.ones(D_obs))

truemodel = HMMSLDS(
    dynamics_distns=dynamics_distns,
    emission_distns=emission_distns,
    init_dynamics_distns=init_dynamics_distns,
    alpha=3., init_state_distn='uniform')

truemodel.mu_init = mu_init
truemodel.sigma_init = sigma_init

#### MANUALLY CREATE DATA
T = 2000
stateseq = np.repeat(np.arange(T//100) % 2, 100).astype(np.int32)
statesobj = truemodel._states_class(model=truemodel, T=stateseq.size, stateseq=stateseq)
statesobj.generate_gaussian_states()
data = statesobj.data = statesobj.generate_obs()
gaussian_states = statesobj.gaussian_states
truemodel.states_list.append(statesobj)

# Mask off a chunk of data
mask = np.ones_like(data, dtype=bool)
chunksz = 200
for i,offset in enumerate(range(0,T,chunksz)):
    j = i % (D_obs + 1)
    if j < D_obs:
        mask[offset:min(offset+chunksz, T), j] = False
    # if j == D_obs:
    #     mask[offset:min(offset+chunksz, T), :] = False

###############
#  make model #
model = HMMSLDS(
    init_dynamics_distns=
        [Gaussian(
            nu_0=5, sigma_0=3.*np.eye(D_latent),
            mu_0=np.zeros(D_latent), kappa_0=0.01,
            mu=mu_init, sigma=sigma_init
        ) for _ in xrange(K)],
    dynamics_distns=
        [AutoRegression(
            A=np.eye(D_latent), sigma=np.eye(D_latent),
            nu_0=D_latent+3,
            S_0=D_latent*np.eye(D_latent),
            M_0=np.zeros((D_latent, D_latent)),
            K_0=D_latent*np.eye(D_latent),
            # A=A, sigma=0.01*np.eye(D_latent)
        ) for A in As],
    emission_distns=DiagonalRegression(D_obs, D_latent,
                                       alpha_0=2.0, beta_0=1.0,
                                       # A=C, sigmasq=0.1*np.ones(D_obs),
                                       ),
    alpha=3., init_state_distn='uniform')
###############
model.add_data(data=data, mask=mask)
# model.trans_distn = copy.deepcopy(truemodel.trans_distn)
# model.states_list[0].stateseq = stateseq.copy()
# model.states_list[0].gaussian_states = gaussian_states.copy()

###############
#  fit model  #
###############
N_samples = 200
def gibbs_update(model):
    model.resample_model()
    return model.log_likelihood(), model.stateseqs[0]

def em_update(model):
    model.EM_step()
    return model.log_likelihood()

def meanfield_update(model):
    model.meanfield_coordinate_descent_step()
    # model.resample_from_mf()
    expected_states = model.states_list[0].expected_states
    return model.log_likelihood(), np.argmax(expected_states, axis=1)

def svi_update(model, stepsize, minibatchsize):
    # Sample a minibatch
    start = np.random.randint(0,T-minibatchsize+1)
    minibatch = data[start:start+minibatchsize]
    minibatch_mask = mask[start:start+minibatchsize]
    prob = minibatchsize/float(T)
    model.meanfield_sgdstep(minibatch, prob, stepsize, masks=minibatch_mask)

    model.resample_from_mf()
    return model.log_likelihood(data)


# Gibbs
# lls, z_smpls = zip(*[gibbs_update(model) for _ in progprint_xrange(N_samples)])

## Mean field
for _ in progprint_xrange(100):
    model.resample_model()
model.states_list[0]._init_mf_from_gibbs()
lls, z_smpls = zip(*[meanfield_update(model) for _ in progprint_xrange(N_samples)])

## SVI
# delay = 10.0
# forgetting_rate = 0.5
# stepsizes = (np.arange(N_samples) + delay)**(-forgetting_rate)
# minibatchsize = 500
# # [model.resample_model() for _ in progprint_xrange(100)]
# lls = [svi_update(model, stepsizes[itr], minibatchsize) for itr in progprint_xrange(N_samples)]


################
# likelihoods  #
################
plt.figure()
plt.plot(lls,'-b')
plt.plot([0,N_samples], truemodel.log_likelihood() * np.ones(2), '-k')
plt.xlabel('iteration')
plt.ylabel('log likelihood')
plt.savefig("log_likelihood.png")


################
#  smoothing   #
################
smoothed_obs = model.states_list[0].smooth()
sample_predictive_obs = model.states_list[0].gaussian_states.dot(model.emission_distns[0].A.T)

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
    plt.plot(smoothed_obs[:,i], 'b', lw=2, label="smoothed")
    # plt.plot(sample_predictive_obs[:,i], ':b', label="sample")

    plt.imshow(1-mask[:,i][None,:],cmap="Greys",alpha=0.25,extent=(0,T) + ylims, aspect="auto")

    if i == 0:
        plt.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.5))

    if i == N_subplots - 1:
        plt.xlabel('time index')

    plt.ylabel("$x_%d(t)$" % (i+1))
    plt.ylim(ylims)
    plt.xlim(xlims)
plt.savefig("slds_missing_data_ex.png")

################
#  z samples   #
################
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(8,3))
gs = gridspec.GridSpec(6,1)
ax1 = fig.add_subplot(gs[:-1])
ax2 = fig.add_subplot(gs[-1], sharex=ax1)

im = ax1.matshow(np.array(z_smpls), aspect='auto')
ax1.autoscale(False)
ax1.set_xticks([])
ax1.set_yticks([])

ax2.matshow(truemodel.stateseqs[0][None,:], aspect='auto')
ax2.set_xticks([])
ax2.set_yticks([])
plt.savefig("slds_discrete_states.png")

plt.show()

