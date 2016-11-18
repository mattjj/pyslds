from __future__ import division
import numpy as np

# NOTE: pass arguments through global variables instead of arguments to exploit
# the fact that they're read-only and multiprocessing/joblib uses fork

model = None
args = None

def _get_sampled_stateseq(idx):
    def resample_states(data_and_kwargs):
        data, kwargs = data_and_kwargs
        model.add_data(data, **kwargs)
        s = model.states_list.pop()
        s.resample()
        return (s.stateseq, s.gaussian_states, s.log_likelihood())

    return map(resample_states, args[idx])
