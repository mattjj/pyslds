import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

# Fancy plotting
try:
    import seaborn as sns
    sns.set_style("white")
    sns.set_context("talk")

    color_names = ["windows blue",
                   "red",
                   "amber",
                   "faded green",
                   "dusty purple",
                   "crimson",
                   "greyish"]
    colors = sns.xkcd_palette(color_names)
except:
    colors = ['b' ,'r', 'y', 'g']

from pybasicbayes.util.text import progprint_xrange
from pylds.util import random_rotation
from pyslds.models import DefaultSLDS, DefaultWeakLimitStickyHDPSLDS

npr.seed(0)

# Set parameters
K = 5
D_obs = 1
D_latent = 2
D_input = 0
T = 2000

# Simulate from one LDS
true_mu_inits = [np.ones(D_latent) for _ in range(K)]
true_sigma_inits = [0.05 * np.eye(D_latent) for _ in range(K)]
# true_As = [random_rotation(D_latent, theta=2*np.pi/(10*(k+1))) for k in range(K)]
true_As = [random_rotation(D_latent) for k in range(K)]
true_sigma_states = [0.05 * np.eye(D_latent) for _ in range(K)]
true_C = np.ones((D_obs, D_latent))
true_sigma_obs = 0.05 * np.eye(D_obs)
true_model = DefaultWeakLimitStickyHDPSLDS(
    K, D_obs, D_latent, D_input=D_input,
    mu_inits=true_mu_inits, sigma_inits=true_sigma_inits,
    As=true_As, sigma_statess=true_sigma_states,
    Cs=true_C, sigma_obss=true_sigma_obs,
    kappa=1000.)
inputs = npr.randn(T, D_input)
y, x, z = true_model.generate(T, inputs=inputs)

# Fit with another LDS
test_model = DefaultSLDS(K, D_obs, D_latent, D_input,
                         Cs=true_C, sigma_obss=true_sigma_obs)
test_model.add_data(y, inputs=inputs)

# Run the Gibbs sampler
N_samples = 1000
def update(model):
    model.resample_model()
    return model.log_likelihood()

lls = [update(test_model) for _ in progprint_xrange(N_samples)]

# Plot the log likelihoods
plt.figure(figsize=(5,3))
plt.plot([0, N_samples], true_model.log_likelihood() * np.ones(2), '--k', label="true")
plt.plot(np.arange(N_samples), lls, color=colors[0], label="test")
plt.xlabel('iteration')
plt.ylabel('training likelihood')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("aux/demo_ll.png")

# Smooth the data
smoothed_data = test_model.smooth(y, inputs)

plt.figure(figsize=(5,3))
plt.plot(y, color=colors[0], lw=2, label="observed")
plt.plot(smoothed_data, color=colors[1], lw=1, label="smoothed")
plt.xlabel("Time")
plt.xlim(0, min(T, 500))
plt.ylabel("Smoothed Data")
plt.ylim(1.2 * np.array(plt.ylim()))
plt.legend(loc="upper center", ncol=2)
plt.tight_layout()
plt.savefig("aux/demo_smooth.png")
plt.show()
