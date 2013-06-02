""" Test model likelihoods
"""

import dismod_mr
import pymc as mc, numpy as np


def test_binom():
    """ Fit a binomial model to a single row of data, confirm the mean
    and std dev are as expected"""

    pi = mc.Uniform('pi', lower=0, upper=1)
    obs = dismod_mr.model.likelihood.binom('prevalence', pi, [.5], [10])

    mc.MCMC([pi, obs]).sample(100000, 50000)
    assert np.allclose(pi.trace().mean(), .5, rtol=.01)
    assert np.allclose(pi.trace().std(), np.sqrt(.5*.5/10.), rtol=.2)  # need to run for a long time to get close on this...

# TODO: add tests for all other likelihoods (beta_binom, poisson, neg_binom, neg_binom_lower_bound, normal, offset_log_normal)
# TODO: add tests for data predicted values
# TODO: change names of p_pred and p_obs
# TODO: add tests for missing data

if __name__ == '__main__':
    import nose
    nose.runmodule()
    
