import numpy as np
np.random.seed(123)

import matplotlib
matplotlib.rcParams.update({'font.sans-serif' : 'Helvetica',
                            'axes.labelsize': 10,
                            'xtick.labelsize' : 6,
                            'ytick.labelsize' : 6,
                            'axes.titlesize' : 10})

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

import seaborn as sns
sns.set(style="white")

color_names = ["red",
               "windows blue",
               "amber",
               "faded green",
               "dusty purple",
               "orange",
               "clay",
               "pink",
               "greyish",
               "light cyan",
               "steel blue",
               "pastel purple",
               "mint",
               "salmon"]

colors = sns.xkcd_palette(color_names)


from pybasicbayes.distributions import Regression, DiagonalRegression, Gaussian

from pyslds.models import HMMSLDS

T = 1000
D = 50
n = T // D

# SLDS
K, D_obs, D_latent = 5, 8, 2


def sample_slds_model():
    mu_init = np.zeros(D_latent)
    mu_init[0] = 2.0
    sigma_init = 0.01 * np.eye(D_latent)

    def random_rotation(n, theta):
        rot = 0.99 * np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
        out = np.zeros((n, n))
        out[:2, :2] = rot
        q = np.linalg.qr(np.random.randn(n, n))[0]
        return q.dot(out).dot(q.T)

    def random_dynamics(n):
        A = np.random.randn(n,n)
        A = A.dot(A.T)
        U,S,V = np.linalg.svd(A)
        A_stable = U.dot(np.diag(S/(1.1*np.max(S)))).dot(V.T)
        # A_stable = U.dot(0.99 * np.eye(n)).dot(V.T)
        return A_stable

    ths = np.linspace(0, np.pi/8., K)
    As = [random_rotation(D_latent, ths[k]) for k in range(K)]
    # As = [random_dynamics(D_latent) for k in range(K)]

    C = np.random.randn(D_obs, D_latent)
    sigma_obs = 0.5 * np.ones(D_obs)

    ###################
    #  generate data  #
    ###################
    init_dynamics_distns = [Gaussian(mu=mu_init, sigma=sigma_init) for _ in range(K)]
    dynamics_distns = [Regression(A=A, sigma=0.01 * np.eye(D_latent)) for A in As]
    emission_distns = DiagonalRegression(D_obs, D_latent, A=C, sigmasq=sigma_obs)

    slds = HMMSLDS(
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns,
        init_dynamics_distns=init_dynamics_distns,
        alpha=3., init_state_distn='uniform')

    #### MANUALLY CREATE DATA
    P = np.ones((K,K)) + 1 * np.eye(K)
    P = P / np.sum(P,1,keepdims=True)
    z = np.zeros(T//D, dtype=np.int32)
    for t in range(1,T//D):
        z[t] = np.random.choice(np.arange(K), p=P[z[t-1]])
    z = np.repeat(z, D)

    statesobj = slds._states_class(model=slds, T=z.size, stateseq=z)
    y = statesobj.data = statesobj.generate_obs()
    x = statesobj.gaussian_states
    slds.states_list.append(statesobj)

    return z,x,y,slds


def draw_slds_figure(z, x, y, filename=None):
    fig = plt.figure(figsize=(5.5, 2.7))
    gs = gridspec.GridSpec(5, 1)
    ax = fig.add_subplot(gs[2:, 0])
    ylim = 1.1 * abs(y).max()
    ymax = ylim * (2*D_obs + 1)
    ymin = -ylim

    # Plot the discrete state in the background
    cps = np.where(np.diff(z) != 0)[0] + 1
    left_cps = np.concatenate(([0], cps))
    right_cps = np.concatenate((cps, [T]))
    for l,r in zip(left_cps, right_cps):
        ax.add_patch(
            Rectangle([l, ymin], r-l, ymax-ymin,
                      color=colors[z[l]], ec="none", alpha=0.5))

        ax.plot([r, r], [ymin, ymax], '-', color="gray", lw=1)

    # Plot the observations
    for i in range(D_obs):
        ax.plot(np.arange(T), ylim * (2*i+1) + y[:,i], '-k', lw=1)

    ax.set_xlim(0, T)
    ax.set_xlabel("time")

    ax.set_ylim(ymin, ymax)
    ax.set_yticks([])
    ax.set_yticks(ylim * (2*np.arange(D_obs)+1))

    def yticklabel(i):
        return "$n=%d$" % (D_obs-i) if (i >= D_obs-2) else \
            "$n=N$" if i == 0 else "."

    ax.set_yticklabels(map(yticklabel, np.arange(D_obs)))
    ax.yaxis.labelpad = 0
    ax.set_ylabel("${\\mathbf{y}_t}$", rotation=0, verticalalignment='center')

    ## Plot the continuous latent state above that above
    ax = fig.add_subplot(gs[1, 0])
    xlim = 1.1 * abs(x).max()
    for l, r in zip(left_cps, right_cps):
        ax.add_patch(
            Rectangle([l, -xlim], r - l, 2*xlim,
                      color=colors[z[l]], ec="none", alpha=0.5))

        ax.plot([r, r], [-xlim, xlim], '-', color="gray", lw=1)

    linestyles = ["-", ":"]
    for i in range(D_latent):
        ax.plot(np.arange(T), x[:, i],
                'k', ls=linestyles[i%len(linestyles)], lw=1)

    ax.set_xlim(0, T)
    ax.set_ylim(-xlim, xlim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.yaxis.labelpad = 34
    ax.set_ylabel("${\\mathbf{x}_t}$", rotation=0, verticalalignment='center')

    ## Now plot the latent state above that above
    ax = fig.add_subplot(gs[0, 0])
    for l, r in zip(left_cps, right_cps):
        ax.add_patch(
            Rectangle([l, 0], r - l, 10,
                      color=colors[z[l]], ec="none", alpha=0.5))

        ax.plot([r, r], [0, 10], '-', color="gray", lw=1)
        ax.text(l + (r - l) / 2. - 10., 3,
                # "u" if z[l] == 0 else "d",
                # string.ascii_lowercase[z[l]],
                z[l]+1,
                fontdict={"size": 9})
    ax.set_xlim(0, T)
    ax.set_ylim(0, 10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.yaxis.labelpad = 34
    ax.set_ylabel("${\\mathbf{z}_t}$", rotation=0, verticalalignment='center')

    if filename is not None:
        fig.savefig(filename, **saveargs)

    plt.show()


def plot_vector_field(k, A, b=None, n_pts=30, xmin=-5, xmax=5,
                      figsize=(5.5/4.0, 5.5/4.0),
                      title="", filename="dynamics_{}"):

    XX, YY = np.meshgrid(
        np.linspace(xmin, xmax, n_pts),
        np.linspace(xmin, xmax, n_pts))
    xx, yy = np.ravel(XX), np.ravel(YY)
    xy = np.column_stack((xx, yy))

    b = 0 if b is None else b
    d_xy = xy.dot(A.T) + b - xy

    # Make the plot
    XY = list(map(np.squeeze, [XX, YY]))
    C = np.ones((n_pts ** 2, 1)) * np.array(colors[k])[None, :]
    C = np.hstack((C, 0.75 * np.ones((n_pts**2, 1))))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, aspect=1.0)
    ax.quiver(XY[0], XY[1], d_xy[:,0], d_xy[:,1], color=C,
              scale=1.0, scale_units="inches",
              headwidth=5.,
              )

    ax.set_xlabel("$x_1$", fontsize=9)
    ax.set_ylabel("$x_2$", fontsize=9)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    plt.tick_params(axis='both', labelsize=8)

    ax.set_title(title, fontsize=10)

if __name__ == "__main__":
    # Sample data
    z,x,y,slds = sample_slds_model()

    # Illustrative figure of SLDS
    draw_slds_figure(z,x,y)

    # Vector fields for latent states
    for k in range(K):
        plot_vector_field(k, slds.dynamics_distns[k].A,
                          title="$A^{(%d)} x_t + b^{(%d)}$" % (k+1,k+1),
                          figsize=(2.75,2.75))
    plt.show()

