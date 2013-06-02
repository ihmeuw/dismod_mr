""" Test model likelihoods
"""

import numpy as np, pymc as mc

import dismod_mr
import data_simulation

def test_data_model_sim():
    # generate simulated data
    data_type = 'p'
    n = 50
    sigma_true = .025
    a = np.arange(0, 100, 1)
    pi_age_true = .0001 * (a * (100. - a) + 100.)

    d = dismod_mr.data.ModelData()
    d.input_data = data_simulation.simulated_age_intervals(data_type, n, a, pi_age_true, sigma_true)
    d.hierarchy, d.output_template = data_simulation.small_output()
    
    # create model and priors
    vars = dismod_mr.model.process.age_specific_rate(d, data_type,
                                 reference_area='all', reference_sex='total', reference_year='all',
                                 mu_age=None, mu_age_parent=None, sigma_age_parent=None)


    # fit model
    m = mc.MCMC(vars)
    m.sample(3)

    # check estimates
    pi_usa = dismod_mr.model.covariates.predict_for(d, d.parameters, 'all', 'total', 'all', 'USA', 'male', 1990, 0., vars[data_type], -np.inf, np.inf)


    # create model w/ emp prior
    # create model and priors
    vars = dismod_mr.model.process.age_specific_rate(d, data_type,
                                 reference_area='all', reference_sex='total', reference_year='all',
                                 mu_age=None, mu_age_parent=pi_usa.mean(0), sigma_age_parent=pi_usa.std(0))


def test_data_model_lower_bound():
    # generate simulated data
    data_type = 'csmr'
    n = 50
    sigma_true = .025
    a = np.arange(0, 100, 1)
    pi_age_true = .0001 * (a * (100. - a) + 100.)

    d = dismod_mr.data.ModelData()
    d.input_data = data_simulation.simulated_age_intervals(data_type, n, a, pi_age_true, sigma_true)
    d.input_data = d.input_data.append(data_simulation.simulated_age_intervals('pf', n, a, pi_age_true*2., sigma_true),
                                       ignore_index=True)
    d.hierarchy, d.output_template = data_simulation.small_output()
    
    # create model and priors
    vars = dismod_mr.model.process.age_specific_rate(d, 'pf',
                                 reference_area='all', reference_sex='total', reference_year='all',
                                 mu_age=None, mu_age_parent=None, sigma_age_parent=None, lower_bound='csmr')


    # fit model
    m = mc.MCMC(vars)
    m.sample(3)

if __name__ == '__main__':
    import nose
    nose.runmodule()
    
