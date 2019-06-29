"""Test Spline Model"""
import numpy as np
import pymc as mc

import dismod_mr


def test_age_pattern_model_sim():
    # simulate normal data
    a = np.arange(0, 100, 5)
    pi_true = .0001 * (a * (100. - a) + 100.)
    sigma_true = .025*np.ones_like(pi_true)

    p = np.maximum(0., mc.rnormal(pi_true, 1./sigma_true**2.))

    # create model and priors
    variables = {}

    variables.update(
        dismod_mr.model.spline.spline('test', ages=np.arange(101), knots=np.arange(0, 101, 5), smoothing=.1))

    variables['pi'] = mc.Lambda('pi', lambda mu=variables['mu_age'], a=a: mu[a])
    variables.update(dismod_mr.model.likelihood.normal('test', variables['pi'], 0., p, sigma_true))

    # fit model
    m = mc.MCMC(variables)
    m.sample(2)
