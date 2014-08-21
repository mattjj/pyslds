from __future__ import division

from pyhsmm.basic.distributions import Regression, Gaussian
from autoregressive.distributions import AutoRegression

###################
#  generate data  #
###################

import autoregressive

As = [np.hstack((-np.eye(2),2*np.eye(2))),
        np.array([[np.cos(np.pi/6),-np.sin(np.pi/6)],[np.sin(np.pi/6),np.cos(np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2))),
        np.array([[np.cos(-np.pi/6),-np.sin(-np.pi/6)],[np.sin(-np.pi/6),np.cos(-np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2)))]

truemodel = autoregressive.models.ARHSMM(
        alpha=4.,init_state_concentration=4.,
        obs_distns=[d.AutoRegression(A=A,sigma=0.1*np.eye(2)) for A in As],
        dur_distns=[pyhsmm.basic.distributions.PoissonDuration(alpha_0=4*25,beta_0=4)
            for state in range(len(As))],
        )

data = truemodel.rvs(1000)

plt.figure()
plt.plot(data[:,0],data[:,1],'bx-')

#################
#  build model  #
#################

# construct distributions
# TODO

# construct model
# TODO

##################
#  run sampling  #
##################

# TODO


