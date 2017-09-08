""" Test model likelihoods
"""

import dismod_mr
import pymc as mc, numpy as np


def test_binom():
    """ Fit a binomial model to a single row of data, confirm the mean
    and std dev are as expected"""

    pi = mc.Uniform('pi', lower=0, upper=1)
    obs = dismod_mr.model.likelihood.binom('prevalence', pi,
                                           np.array([.5]),
                                           np.array([10]))

    mc.MAP([pi, obs]).fit()
    assert np.allclose(pi.value, .5, rtol=.01)

    #mc.MCMC([pi, obs]).sample(100000, 50000)
    #assert np.allclose(pi.trace().mean(), .5, rtol=.01)
    #assert np.allclose(pi.trace().std(), np.sqrt(.5*.5/10.), rtol=.2)  # need to run for a long time to get close on this...

def test_beta_binom_2():
    """ Fit a fast beta binomial model to a single row of data, confirm the mean
    and std dev are as expected"""

    pi = mc.Uniform('pi', lower=0, upper=1, size=2)
    obs = dismod_mr.model.likelihood.beta_binom_2('prevalence', pi, 
                                                  np.array([1,1]), np.array([.5,.5]), np.array([10,10]))

    mc.MAP([pi, obs]).fit()
    assert np.allclose(pi.value, .5, rtol=.01)

    #mc.MCMC([pi, obs]).sample(100000, 50000)
    #assert np.allclose(pi.trace().mean(), .5, rtol=.01)
    #assert np.allclose(pi.trace().std(), np.sqrt(.5*.5/10.), rtol=.2)  # need to run for a long time to get close on this...


def test_normal():
    """ Fit a normal model to a single row of data, confirm the mean
    and std dev are as expected"""

    pi = mc.Uniform('pi', lower=0, upper=1)
    sigma = mc.Uniform('sigma', lower=0, upper=1)
    obs = dismod_mr.model.likelihood.normal('prevalence', pi, sigma, [.5], [.1])

    mc.MAP([pi, sigma, obs]).fit()
    assert np.allclose(pi.value, .5, rtol=.01)

# TODO: add tests for all other likelihoods (beta_binom, poisson, neg_binom, neg_binom_lower_bound, offset_log_normal)
# TODO: add tests for data predicted values
# TODO: change names of p_pred and p_obs
# TODO: add tests for missing data

if __name__ == '__main__':
    import nose
    nose.runmodule()
    
