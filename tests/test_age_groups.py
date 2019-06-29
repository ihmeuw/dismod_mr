""" Test age group models
"""
import data_simulation
import numpy as np, pymc as mc

# TODO: test that this works when ages includes non-integral values
# (requires getting age weights right)

def test_age_standardizing_approx():
    # simulate data
    n = 50
    sigma_true = .025*np.ones(n)
    a = np.arange(0, 100, 1)
    pi_age_true = .0001 * (a * (100. - a) + 100.)
    ages=np.arange(101)
    d = data_simulation.simulated_age_intervals('p', n, a, pi_age_true, sigma_true)

    # create model and priors
    vars = {}
    vars.update(src.dismod_mr.model.spline.spline('test', ages, knots=np.arange(0, 101, 5), smoothing=.01))
    vars.update(src.dismod_mr.model.age_groups.age_standardize_approx('test', np.ones_like(vars['mu_age'].value), vars['mu_age'], d['age_start'], d['age_end'], ages))
    vars['pi'] = vars['mu_interval']
    vars.update(src.dismod_mr.model.likelihood.normal('test', pi=vars['pi'], sigma=0, p=d['value'], s=sigma_true))

    # fit model
    m = mc.MCMC(vars)
    m.sample(3)

def test_age_integrating_midpoint_approx():
    # simulate data
    n = 50
    sigma_true = .025*np.ones(n)
    a = np.arange(0, 100, 1)
    pi_age_true = .0001 * (a * (100. - a) + 100.)
    ages = np.arange(101)
    d = data_simulation.simulated_age_intervals('p', n, a, pi_age_true, sigma_true)

    # create model and priors
    vars = {}
    vars.update(src.dismod_mr.model.spline.spline('test', ages, knots=np.arange(0, 101, 5), smoothing=.01))
    vars.update(
        src.dismod_mr.model.age_groups.midpoint_approx('test', vars['mu_age'], d['age_start'], d['age_end'], ages))
    vars['pi'] = vars['mu_interval']
    vars.update(src.dismod_mr.model.likelihood.normal('test', pi=vars['pi'], sigma=0, p=d['value'], s=sigma_true))

    # fit model
    m = mc.MCMC(vars)
    m.sample(3)

if __name__ == '__main__':
    import nose
    nose.runmodule()

