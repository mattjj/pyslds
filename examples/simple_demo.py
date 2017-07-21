import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Fancy plotting
try:
    import seaborn as sns
    from hips.plotting.colormaps import gradient_cmap
    sns.set_style("white")
    sns.set_context("paper")

    color_names = ["red",
                   "windows blue",
                   "medium green",
                   "dusty purple",
                   "orange",
                   "amber",
                   "clay",
                   "pink",
                   "greyish",
                   "light cyan",
                   "steel blue",
                   "forest green",
                   "pastel purple",
                   "mint",
                   "salmon",
                   "dark brown"]
    colors = sns.xkcd_palette(color_names)
    cmap = gradient_cmap(colors)
except:
    from matplotlib.cm import get_cmap
    colors = ['b', 'r', 'y', 'g']
    cmap = get_cmap("jet")


from pybasicbayes.util.text import progprint_xrange
from pylds.util import random_rotation
from pyslds.models import DefaultSLDS, DefaultWeakLimitStickyHDPSLDS

npr.seed(0)

# Set parameters
K = 5
D_obs = 10
D_latent = 2
D_input = 1
T = 1000

# Make an LDS with known parameters
true_mu_inits = [np.ones(D_latent) for _ in range(K)]
true_sigma_inits = [np.eye(D_latent) for _ in range(K)]
true_As = [.99 * random_rotation(D_latent, theta=np.pi/((k+1) * 4)) for k in range(K)]
true_Bs = [3 * npr.randn(D_latent, D_input) for k in range(K)]
true_sigma_states = [np.eye(D_latent) for _ in range(K)]
true_C = np.random.randn(D_obs, D_latent)
true_Ds = np.zeros((D_obs, D_input))
true_sigma_obs = 0.05 * np.eye(D_obs)
true_model = DefaultSLDS(
    K, D_obs, D_latent, D_input=D_input,
    mu_inits=true_mu_inits, sigma_inits=true_sigma_inits,
    As=true_As, Bs=true_Bs, sigma_statess=true_sigma_states,
    Cs=true_C, Ds=true_Ds, sigma_obss=true_sigma_obs)

# Simulate some data with a given discrete state sequence
inputs = np.ones((T, D_input))
z = np.arange(K).repeat(T // K)
y, x, _ = true_model.generate(T, inputs=inputs, stateseq=z)

# Fit with another LDS.  Give it twice as many states in
# order to have some flexibility during inference.
test_model = DefaultSLDS(2*K, D_obs, D_latent, D_input,
                         Cs=npr.randn(D_obs, D_latent),
                         Ds=npr.randn(D_obs, D_input))
test_model.add_data(y, inputs=inputs)

# Initialize with Gibbs sampler
print("Initializing with Gibbs")
N_gibbs_samples = 100
def initialize(model):
    model.resample_model()
    return model.log_likelihood()

gibbs_lls = [initialize(test_model) for _ in progprint_xrange(N_gibbs_samples)]

# Fit with VBEM
print("Fitting with VBEM")
N_vbem_iters = 100
def update(model):
    model.VBEM_step()
    return model.log_likelihood()

test_model.states_list[0]._init_mf_from_gibbs()
vbem_lls = [update(test_model) for _ in progprint_xrange(N_vbem_iters)]

# Plot the log likelihoods
plt.figure(figsize=(5,3))
plt.plot([0, N_gibbs_samples + N_vbem_iters], true_model.log_likelihood() * np.ones(2), '--k', label="true")
plt.plot(np.arange(N_gibbs_samples), gibbs_lls, color=colors[0], label="gibbs")
plt.plot(np.arange(N_gibbs_samples + 1, N_gibbs_samples + N_vbem_iters), vbem_lls[1:], color=colors[1], label="vbem")
plt.xlim(0, N_gibbs_samples + N_vbem_iters)
plt.xlabel('iteration')
plt.ylabel('log likelihood')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("aux/demo_ll.png")

# Smooth the data
smoothed_data = test_model.smooth(y, inputs)

fig = plt.figure(figsize=(5,3))
gs = GridSpec(3, 1, height_ratios=[.1, .1, 1.0])
ax = fig.add_subplot(gs[0,0])
ax.imshow(true_model.states_list[0].stateseq[None,:], vmin=0, vmax=len(colors)-1,
          cmap=cmap, interpolation="nearest", aspect="auto")
ax.set_xticklabels([])
ax.set_yticks([])
ax.set_title("True Discrete States")

ax = fig.add_subplot(gs[1,0])
ax.imshow(test_model.states_list[0].stateseq[None,:], vmin=0, vmax=len(colors)-1,
          cmap=cmap, interpolation="nearest", aspect="auto")
ax.set_xticklabels([])
ax.set_yticks([])
ax.set_title("Inferred Discrete States")

ax = fig.add_subplot(gs[2,0])
plt.plot(y[:,0], color='k', lw=2, label="observed")
plt.plot(smoothed_data[:,0], color=colors[0], lw=1, label="smoothed")
plt.xlabel("Time")
plt.xlim(0, min(T, 500))
plt.ylabel("Observations")
plt.legend(loc="upper center", ncol=2)
plt.tight_layout()
plt.savefig("aux/demo_smooth.png")

plt.figure()
from pyhsmm.util.general import rle
z_rle = rle(z)
offset = 0
for k, dur in zip(*z_rle):
    plt.plot(x[offset:offset+dur,0], x[offset:offset+dur,1], color=colors[k])
    offset += dur

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Continuous Latent States")
plt.show()
