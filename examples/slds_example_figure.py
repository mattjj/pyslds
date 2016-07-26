import string
import numpy as np
np.random.seed(123)

from scipy.misc import logsumexp

import matplotlib
matplotlib.rcParams.update({'font.sans-serif' : 'Helvetica',
                            'axes.labelsize': 10,
                            'xtick.labelsize' : 6,
                            'ytick.labelsize' : 6,
                            'axes.titlesize' : 10})

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from hips.plotting.layout import create_figure, create_axis_at_location


import seaborn as sns
color_names = ["windows blue",
               "amber",
               "faded green",
               "dusty purple",
               "crimson",
               "greyish"]
colors = sns.xkcd_palette(color_names)
sns.set(style="white", palette=sns.xkcd_palette(color_names))


from hips.plotting.colormaps import harvard_colors, gradient_cmap
#colors = harvard_colors()


from pybasicbayes.distributions import AutoRegression, DiagonalRegression, Gaussian

from pyslds.models import HMMSLDS

T = 1000
D = 50
n = T // D

# SLDS
K, D_obs, D_latent = 4, 8, 2

def sample_mixture_model(lmbda, p):
    """
    Simple mixture model example
    """
    # Simulate latent states
    z = np.random.rand(n) < p
    
    # Simulate real valued spike times
    Ss = []
    Ns = np.zeros(n)
    for i in np.arange(n):
        rate = lmbda[z[i]]
        Ns[i] = np.random.poisson(rate * 0.05)
        Ss.append(i * D + np.random.rand(Ns[i]) * D)

    Ss = np.concatenate(Ss)

    return Ns, Ss, z

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
    init_dynamics_distns = [Gaussian(mu=mu_init, sigma=sigma_init) for _ in xrange(K)]
    dynamics_distns = [AutoRegression(A=A, sigma=0.01 * np.eye(D_latent)) for A in As]
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

    # z = np.repeat(np.random.randint(0,K,size=(T//D)), D).astype(np.int32)
    # z = np.repeat(np.arange(T // D) % K, D).astype(np.int32)
    statesobj = slds._states_class(model=slds, T=z.size, stateseq=z)
    statesobj.generate_gaussian_states()
    y = statesobj.data = statesobj.generate_obs()
    x = statesobj.gaussian_states
    slds.states_list.append(statesobj)

    return z,x,y,slds


def draw_mixture_figure(Ns, Ss, z, lmbda,
                        filename="mixture.png",
                        saveargs=dict(dpi=300),
                        show=False):
    fig = create_figure((5.5, 2.7))
    ax = create_axis_at_location(fig, .75, .5, 4., 1.375)
    ymax = 105
    # Plot the rates
    for i in range(n):
        ax.add_patch(Rectangle([i*D,0], D, lmbda[z[i]],
                               color=colors[z[i]], ec="none", alpha=0.5))
        ax.plot([i*D, (i+1)*D], lmbda[z[i]] * np.ones(2), '-k', lw=2)

        if i < n-1:
            ax.plot([(i+1)*D, (i+1)*D], [lmbda[z[i]], lmbda[z[i+1]]], '-k', lw=2)
            
        # Plot boundaries
        ax.plot([(i+1)*D, (i+1)*D], [0, ymax], ':k', lw=1)
        
        
    # Plot x axis
    plt.plot([0,T], [0,0], '-k', lw=2)

    # Plot spike times
    for s in Ss:
        plt.plot([s,s], [0,60], '-ko', markerfacecolor='k', markersize=5)

    plt.xlabel("time [ms]")
    plt.ylabel("firing rate [Hz]")
    plt.xlim(0,T)
    plt.ylim(-5,ymax)

    ## Now plot the spike count above
    ax = create_axis_at_location(fig, .75, 2., 4., .25)
    for i in xrange(n):
        # Plot boundaries
        ax.plot([(i+1)*D, (i+1)*D], [0, 10], '-k', lw=1)
        ax.text(i*D + D/3.5, 3, "%d" % Ns[i], fontdict={"size":9})
    ax.set_xlim(0,T)
    ax.set_ylim(0,10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.yaxis.labelpad = 30
    ax.set_ylabel("${s_t}$", rotation=0,  verticalalignment='center')

    ## Now plot the latent state above that above
    ax = create_axis_at_location(fig, .75, 2.375, 4., .25)
    for i in xrange(n):
        # Plot boundaries
        ax.add_patch(Rectangle([i*D,0], D, 10,
                            color=colors[z[i]], ec="none", alpha=0.5))

        ax.plot([(i+1)*D, (i+1)*D], [0, 10], '-k', lw=1)
        ax.text(i*D + D/3.5, 3, "u" if z[i]==0 else "d", fontdict={"size":9})
    ax.set_xlim(0,T)
    ax.set_ylim(0,10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.yaxis.labelpad = 30
    ax.set_ylabel("${z_t}$", rotation=0,  verticalalignment='center')

    
    #fig.savefig(filename + ".pdf")
    fig.savefig(filename, **saveargs)
    if show:
        plt.show()
    else:
        plt.close(fig)


def draw_slds_figure(z, x, y, slds, filename="slds", saveargs=dict(dpi=300)):
    fig = create_figure((5.5, 2.7))
    ax = create_axis_at_location(fig, .75, .5, 4., 1.375)
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

    # ax.set_ylabel("observations")
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
    ax = create_axis_at_location(fig, .75, 2., 4., .25)
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
    ax = create_axis_at_location(fig, .75, 2.375, 4., .25)
    for l, r in zip(left_cps, right_cps):
        ax.add_patch(
            Rectangle([l, 0], r - l, 10,
                      color=colors[z[l]], ec="none", alpha=0.5))

        ax.plot([r, r], [0, 10], '-', color="gray", lw=1)
        ax.text(l + (r - l) / 2. - 10., 3,
                # "u" if z[l] == 0 else "d",
                string.ascii_lowercase[z[l]],
                fontdict={"size": 9})
    ax.set_xlim(0, T)
    ax.set_ylim(0, 10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.yaxis.labelpad = 34
    ax.set_ylabel("${\\mathbf{z}_t}$", rotation=0, verticalalignment='center')

    # fig.savefig(filename + ".pdf")
    fig.savefig(filename+".pdf", **saveargs)
    fig.savefig(filename+".png", **saveargs)

    # plt.close(fig)
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
    XY = map(np.squeeze, [XX, YY])
    C = np.ones((n_pts ** 2, 1)) * np.array(colors[k])[None, :]
    C = np.hstack((C, 0.75 * np.ones((n_pts**2, 1))))

    fig = plt.figure(figsize=figsize)
    # ax = fig.add_subplot(111, aspect=1.0)
    width = min(*figsize) - .75
    ax = create_axis_at_location(fig, .5, .5, width, width)
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
    # plt.tight_layout()

    fig.savefig(filename.format(k+1) + ".pdf")
    fig.savefig(filename.format(k+1) + ".png")

def make_mixture_mcmc_figures(Ns, Ss, z0, lmbda0, p=0.5, a_lmbda=1, b_lmbda=1, N_iter=100):

    def _poisson_ll(s, l):
        return -l + s*np.log(l)
    
    def _resample():
        # Resample latent states given lmbda
        for i in xrange(n):
            lp0 = _poisson_ll(Ns[i], lmbda[0]) + np.log(p)
            lp1 = _poisson_ll(Ns[i], lmbda[1]) + np.log(1-p)
            p_0 = np.exp(lp0 - logsumexp([lp0, lp1]))
            z[i] = np.random.rand() < 1-p_0

        # Resample lmbda given z
        for k in [0,1]:
            Nk = (z==k).sum()
            Sk = Ns[z==k].sum()
            a_post = a_lmbda + Sk
            b_post = b_lmbda + Nk
            lmbda[k] = np.random.gamma(a_post, 1./b_post)

    # Now run the Gibbs sampler and save out images
    lmbda = lmbda0.copy()
    z = z0.copy()
    for itr in range(N_iter):
        print "Iteration ", itr
        draw_mixture_figure(Ns, Ss, z, lmbda/0.05, filename="itr_%d.jpg" % itr)
        _resample()
                        
if __name__ == "__main__":
    # Sample data
    z,x,y,slds = sample_slds_model()
    draw_slds_figure(z,x,y,slds)

    for k in range(K):
        plot_vector_field(k, slds.dynamics_distns[k].A,
                          title="$A^{(%d)} x_t + b^{(%d)}$" % (k+1,k+1),
                          figsize=(2.75,2.75))
    plt.show()

    # Sample data
    # lmbda = np.array([100, 10])
    # p = 0.5
    # Ns, Ss, z = sample_mixture_model(lmbda, p)
    # draw_mixture_figure(Ns, Ss, z, lmbda)
    
    #z0 = np.random.rand(n) < 0.5
    #z0 = np.zeros(n, dtype=np.bool)
    #lmbda0 = np.random.gamma(1,1,size=2)
    #lmbda0 = 1 * np.ones(2)
    #make_mcmc_figures(Ns, Ss, z0, lmbda0)
