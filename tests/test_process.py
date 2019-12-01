"""Test model likelihoods."""
import numpy as np
import pandas as pd
import pymc as mc

import dismod_mr
from dismod_mr.testing import data_simulation

# TODO: add test of consistent model, confirm that is working


def test_age_specific_rate_model():
    # generate simulated data
    data_type = 'p'
    n = 50
    sigma_true = .025
    a = np.arange(0, 100, 1)
    pi_age_true = .0001 * (a * (100. - a) + 100.)

    d = dismod_mr.data.ModelData()
    d.add_data(data_simulation.simulated_age_intervals(data_type, n, a, pi_age_true, sigma_true))
    d.hierarchy, d.output_template = data_simulation.small_output()

    # create model and priors
    variables = dismod_mr.model.process.age_specific_rate(d, data_type,
                                                          reference_area='all',
                                                          reference_sex='total',
                                                          reference_year='all',
                                                          mu_age=None,
                                                          mu_age_parent=None,
                                                          sigma_age_parent=None)

    # fit model
    m = mc.MCMC(variables)
    m.sample(3)

    # check estimates
    pi_usa = dismod_mr.model.covariates.predict_for(d, d.parameters, 'all', 'total', 'all', 'USA', 'male',
                                                    1990, 0., variables[data_type], -np.inf, np.inf)

    # create model w/ emp prior
    # create model and priors
    variables = dismod_mr.model.process.age_specific_rate(d, data_type,
                                                          reference_area='all',
                                                          reference_sex='total',
                                                          reference_year='all',
                                                          mu_age=None,
                                                          mu_age_parent=pi_usa.mean(0),
                                                          sigma_age_parent=pi_usa.std(0))


def test_age_specific_rate_model_w_lower_bound_data():
    # generate simulated data
    data_type = 'csmr'
    n = 50
    sigma_true = .025
    a = np.arange(0, 100, 1)
    pi_age_true = .0001 * (a * (100. - a) + 100.)

    d = dismod_mr.data.ModelData()
    d.add_data(data_simulation.simulated_age_intervals(data_type, n, a, pi_age_true, sigma_true))
    d.add_data(data_simulation.simulated_age_intervals('pf', n, a, pi_age_true*2., sigma_true))
    d.hierarchy, d.output_template = data_simulation.small_output()

    # create model and priors
    variables = dismod_mr.model.process.age_specific_rate(d, 'pf',
                                                          reference_area='all',
                                                          reference_sex='total',
                                                          reference_year='all',
                                                          mu_age=None,
                                                          mu_age_parent=None,
                                                          sigma_age_parent=None,
                                                          lower_bound='csmr')

    # fit model
    m = mc.MCMC(variables)
    m.sample(3)


def test_consistent():
    np.random.seed(123456)
    dm = dismod_mr.data.ModelData()
    dm.hierarchy, dm.output_template = data_simulation.small_output()

    # create model and priors
    dm.vars = dismod_mr.model.process.consistent(dm)

    mc.MCMC(dm.vars).sample(3)

    # try it again with expert priors on prevalence

    dm.parameters['p']['level_value'] = dict(value=.1, age_before=15, age_after=95)
    dm.parameters['p']['level_bounds'] = dict(upper=.01, lower=.001)

    # create model and priors
    dm.vars = dismod_mr.model.process.consistent(dm)

    mc.MCMC(dm.vars).sample(3)


def test_consistent_w_non_integral_ages():
    np.random.seed(1234567)
    dm = dismod_mr.data.ModelData()
    dm.hierarchy, dm.output_template = data_simulation.small_output()

    # change to non-integral ages
    dm.parameters['ages'] = np.arange(0., 5.1, .1)
    for k in dm.parameters:
        if type(dm.parameters[k]) == dict:
            dm.parameters[k]['parameter_age_mesh'] = [0,5]

    # create model and priors
    dm.vars = dismod_mr.model.process.consistent(dm)

    mc.MCMC(dm.vars).sample(3)

    # try again, this time with m_all data
    m_all = (pd.DataFrame([['m_all', 0, 100, .1]], columns=['data_type', 'age_start', 'age_end', 'value']))
    m_all = data_simulation.add_standard_columns(m_all)
    dm.add_data(m_all)
    dm.vars = dismod_mr.model.process.consistent(dm)

    mc.MCMC(dm.vars).sample(3)
