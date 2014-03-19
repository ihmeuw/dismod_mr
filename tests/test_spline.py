""" Test Spline Model
"""

import dismod_mr

import numpy as np,  pymc as mc

def test_age_pattern_model_sim():
    # simulate normal data
    a = np.arange(0, 100, 5)
    pi_true = .0001 * (a * (100. - a) + 100.)
    sigma_true = .025*np.ones_like(pi_true)

    p = np.maximum(0., mc.rnormal(pi_true, 1./sigma_true**2.))

    # create model and priors
    vars = {}

    vars.update(dismod_mr.model.spline.spline('test', ages=np.arange(101), knots=np.arange(0,101,5), smoothing=.1))

    vars['pi'] = mc.Lambda('pi', lambda mu=vars['mu_age'], a=a: mu[a])
    vars.update(dismod_mr.model.likelihood.normal('test', vars['pi'], 0., p, sigma_true))

    # fit model
    m = mc.MCMC(vars)
    m.sample(2)

# TODO: test that linear interpolation works as expected with 2 knot spline
# TODO: test that smoothing works as expected

if __name__ == '__main__':
    import nose
    nose.runmodule()
    
