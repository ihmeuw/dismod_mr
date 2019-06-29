"""Test expert priors."""
import numpy as np
import pymc as mc

import dismod_mr

# TODO: add test of covariate level constraint


def test_prior_level_constraint():
    d = dismod_mr.data.ModelData()
    ages = np.arange(101)

    # create model with no priors
    variables = {}
    variables.update(dismod_mr.model.spline.spline('test', ages, knots=np.arange(0, 101, 5), smoothing=.01))
    variables.update(dismod_mr.model.priors.level_constraints('test', {}, variables['mu_age'], ages))

    # fit model
    m = mc.MCMC(variables)
    m.sample(3)

    # create model with expert priors
    parameters = {'level_value': dict(value=.1, age_before=15, age_after=95),
                  'level_bounds': dict(upper=.01, lower=.001)}

    variables = {}
    variables.update(dismod_mr.model.spline.spline('test', ages, knots=np.arange(0, 101, 5), smoothing=.01))
    variables.update(dismod_mr.model.priors.level_constraints('test', parameters, variables['mu_age'], ages))

    assert 'mu_sim' in variables

    # fit model
    m = mc.MCMC(variables)
    m.sample(3)


def test_prior_derivative_sign():
    d = dismod_mr.data.ModelData()
    ages = np.arange(101)

    # create model with no priors
    variables = {}
    variables.update(dismod_mr.model.spline.spline('test', ages, knots=np.arange(0, 101, 5), smoothing=.01))
    variables.update(dismod_mr.model.priors.derivative_constraints('test', {}, variables['mu_age'], ages))

    # create model with expert priors
    parameters = {'increasing': dict(age_start=15, age_end=95),
                  'decreasing': dict(age_start=0, age_end=0)}
    variables = {}
    variables.update(dismod_mr.model.spline.spline('test', ages, knots=np.arange(0, 101, 5), smoothing=.01))
    variables.update(dismod_mr.model.priors.derivative_constraints('test', parameters, variables['mu_age'],
                                                                   variables['knots']))

    # fit model
    m = mc.MCMC(variables)
    m.sample(3)
