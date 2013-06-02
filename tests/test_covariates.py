""" Test models that use covariates
"""

import numpy as np, pymc as mc, pandas as pd
import dismod_mr
import data_simulation

def test_covariate_model_sim_no_hierarchy():
    # simulate normal data
    model = dismod_mr.data.ModelData()
    model.hierarchy, model.output_template = data_simulation.small_output()

    X = mc.rnormal(0., 1.**2, size=(128,3))

    beta_true = [-.1, .1, .2]
    Y_true = np.dot(X, beta_true)

    pi_true = np.exp(Y_true)
    sigma_true = .01*np.ones_like(pi_true)

    p = mc.rnormal(pi_true, 1./sigma_true**2.)

    model.input_data = pd.DataFrame(dict(value=p, x_0=X[:,0], x_1=X[:,1], x_2=X[:,2]))
    model.input_data['area'] = 'all'
    model.input_data['sex'] = 'total'
    model.input_data['year_start'] = 2000
    model.input_data['year_end'] = 2000

    # create model and priors
    vars = {}
    vars.update(dismod_mr.model.covariates.mean_covariate_model('test', 1, model.input_data, {}, model, 'all', 'total', 'all'))
    vars.update(dismod_mr.model.likelihood.normal('test', vars['pi'], 0., p, sigma_true))

    # fit model
    m = mc.MCMC(vars)
    m.sample(2)


def test_covariate_model_sim_w_hierarchy():
    n = 50

    # setup hierarchy
    hierarchy, output_template = data_simulation.small_output()

    # simulate normal data
    area_list = np.array(['all', 'USA', 'CAN'])
    area = area_list[mc.rcategorical([.3, .3, .4], n)]

    sex_list = np.array(['male', 'female', 'total'])
    sex = sex_list[mc.rcategorical([.3, .3, .4], n)]

    year = np.array(mc.runiform(1990, 2010, n), dtype=int)
        
    alpha_true = dict(all=0., USA=.1, CAN=-.2)

    pi_true = np.exp([alpha_true[a] for a in area])
    sigma_true = .05*np.ones_like(pi_true)

    p = mc.rnormal(pi_true, 1./sigma_true**2.)

    model = dismod_mr.data.ModelData()
    model.input_data = pd.DataFrame(dict(value=p, area=area, sex=sex, year_start=year, year_end=year))
    model.hierarchy, model.output_template = hierarchy, output_template

    # create model and priors
    vars = {}
    vars.update(dismod_mr.model.covariates.mean_covariate_model('test', 1, model.input_data, {}, model,
                                                     'all', 'total', 'all'))
    vars.update(dismod_mr.model.likelihood.normal('test', vars['pi'], 0., p, sigma_true))

    # fit model
    m = mc.MCMC(vars)
    m.sample(2)

    assert 'sex' not in vars['U']
    assert 'x_sex' in vars['X']
    assert len(vars['beta']) == 1

def test_fixed_effect_priors():
    model = dismod_mr.data.ModelData()

    # set prior on sex
    parameters = dict(fixed_effects={'x_sex': dict(dist='TruncatedNormal', mu=1., sigma=.5, lower=-10, upper=10)})

    # simulate normal data
    n = 32.
    sex_list = np.array(['male', 'female', 'total'])
    sex = sex_list[mc.rcategorical([.3, .3, .4], n)]
    beta_true = dict(male=-1., total=0., female=1.)
    pi_true = np.exp([beta_true[s] for s in sex])
    sigma_true = .05
    p = mc.rnormal(pi_true, 1./sigma_true**2.)

    model.input_data = pd.DataFrame(dict(value=p, sex=sex))
    model.input_data['area'] = 'all'
    model.input_data['year_start'] = 2010
    model.input_data['year_start'] = 2010



    # create model and priors
    vars = {}
    vars.update(dismod_mr.model.covariates.mean_covariate_model('test', 1, model.input_data, parameters, model,
                                                     'all', 'total', 'all'))

    print vars['beta']
    assert vars['beta'][0].parents['mu'] == 1.

def test_random_effect_priors():
    model = dismod_mr.data.ModelData()

    # set prior on sex
    parameters = dict(random_effects={'USA': dict(dist='TruncatedNormal', mu=.1, sigma=.5, lower=-10, upper=10)})


    # simulate normal data
    n = 32.
    area_list = np.array(['all', 'USA', 'CAN'])
    area = area_list[mc.rcategorical([.3, .3, .4], n)]
    alpha_true = dict(all=0., USA=.1, CAN=-.2)
    pi_true = np.exp([alpha_true[a] for a in area])
    sigma_true = .05
    p = mc.rnormal(pi_true, 1./sigma_true**2.)

    model.input_data = pd.DataFrame(dict(value=p, area=area))
    model.input_data['sex'] = 'male'
    model.input_data['year_start'] = 2010
    model.input_data['year_end'] = 2010

    model.hierarchy.add_edge('all', 'USA')
    model.hierarchy.add_edge('all', 'CAN')

    # create model and priors
    vars = {}
    vars.update(dismod_mr.model.covariates.mean_covariate_model('test', 1, model.input_data, parameters, model,
                                                     'all', 'total', 'all'))

    print vars['alpha']
    print vars['alpha'][1].parents['mu']
    assert vars['alpha'][1].parents['mu'] == .1


def test_covariate_model_dispersion():
    # simulate normal data
    n = 100

    model = dismod_mr.data.ModelData()
    model.hierarchy, model.output_template = data_simulation.small_output()

    Z = mc.rcategorical([.5, 5.], n)
    zeta_true = -.2

    pi_true = .1
    ess = 10000.*np.ones(n)
    eta_true = np.log(50)
    delta_true = 50 + np.exp(eta_true)

    p = mc.rnegative_binomial(pi_true*ess, delta_true*np.exp(Z*zeta_true)) / ess

    
    model.input_data = pd.DataFrame(dict(value=p, z_0=Z))
    model.input_data['area'] = 'all'
    model.input_data['sex'] = 'total'
    model.input_data['year_start'] = 2000
    model.input_data['year_end'] = 2000



    # create model and priors
    vars = dict(mu=mc.Uninformative('mu_test', value=pi_true))
    vars.update(dismod_mr.model.covariates.mean_covariate_model('test', vars['mu'], model.input_data, {}, model, 'all', 'total', 'all'))
    vars.update(dismod_mr.model.covariates.dispersion_covariate_model('test', model.input_data, .1, 10.))
    vars.update(dismod_mr.model.likelihood.neg_binom('test', vars['pi'], vars['delta'], p, ess))

    # fit model
    m = mc.MCMC(vars)
    m.sample(2)

def test_covariate_model_shift_for_root_consistency():
    # generate simulated data
    n = 50
    sigma_true = .025
    a = np.arange(0, 100, 1)
    pi_age_true = .0001 * (a * (100. - a) + 100.)
    
    d = dismod_mr.data.ModelData()
    d.input_data = data_simulation.simulated_age_intervals('p', n, a, pi_age_true, sigma_true)
    d.hierarchy, d.output_template = data_simulation.small_output()
    

    # create model and priors
    vars = dismod_mr.model.process.age_specific_rate(d, 'p', 'all', 'total', 'all', None, None, None)

    vars = dismod_mr.model.process.age_specific_rate(d, 'p', 'all', 'male', 1990, None, None, None)

    # fit model
    m = mc.MCMC(vars)

    m.sample(3)

    # check estimates
    pi_usa = dismod_mr.model.covariates.predict_for(d, d.parameters['p'], 'all', 'male', 1990, 'USA', 'male', 1990, 0., vars['p'], 0., np.inf)

def test_predict_for():
    """ Approach to testing predict_for function:

    1. Create model with known mu_age, known covariate values, known effect coefficients
    2. Setup MCMC with NoStepper for all stochs
    3. Sample to generate trace with known values
    4. Predict for results, and confirm that they match expected values
    """
    
    # generate simulated data
    n = 5
    sigma_true = .025
    a = np.arange(0, 100, 1)
    pi_age_true = .0001 * (a * (100. - a) + 100.)
    
    d = dismod_mr.data.ModelData()
    d.input_data = data_simulation.simulated_age_intervals('p', n, a, pi_age_true, sigma_true)
    d.hierarchy, d.output_template = data_simulation.small_output()


    # create model and priors
    vars = dismod_mr.model.process.age_specific_rate(d, 'p', 'all', 'total', 'all', None, None, None)

    # fit model
    m = mc.MCMC(vars)
    for n in m.stochastics:
        m.use_step_method(mc.NoStepper, n)
    m.sample(3)


    ### Prediction case 1: constant zero random effects, zero fixed effect coefficients

    # check estimates with priors on random effects
    d.parameters['p']['random_effects'] = {}
    for node in ['USA', 'CAN', 'NAHI', 'super-region-1', 'all']:
        d.parameters['p']['random_effects'][node] = dict(dist='Constant', mu=0, sigma=1.e-9) # zero out REs to see if test passes
        
    pred = dismod_mr.model.covariates.predict_for(d, d.parameters['p'],
                                         'all', 'total', 'all',
                                         'USA', 'male', 1990,
                                         0., vars['p'], 0., np.inf)

    # test that the predicted value is as expected
    fe_usa_1990 = 1.
    re_usa_1990 = 1.
    assert_almost_equal(pred,
                        vars['p']['mu_age'].trace() * fe_usa_1990 * re_usa_1990)


    ### Prediction case 2: constant non-zero random effects, zero fixed effect coefficients

    # check estimates with priors on random effects
    for i, node in enumerate(['USA', 'NAHI', 'super-region-1']):
        d.parameters['p']['random_effects'][node]['mu'] = (i+1.)/10.
        
    pred = dismod_mr.model.covariates.predict_for(d, d.parameters['p'],
                                         'all', 'total', 'all',
                                         'USA', 'male', 1990,
                                         0., vars['p'], 0., np.inf)

    # test that the predicted value is as expected
    fe_usa_1990 = 1.
    re_usa_1990 = np.exp(.1+.2+.3)
    assert_almost_equal(pred,
                        vars['p']['mu_age'].trace() * fe_usa_1990 * re_usa_1990)


    ### Prediction case 3: confirm that changing RE for reference area does not change results

    d.parameters['p']['random_effects']['all']['mu'] = 1.
        
    pred = dismod_mr.model.covariates.predict_for(d, d.parameters['p'],
                                         'all', 'total', 'all',
                                         'USA', 'male', 1990,
                                         0., vars['p'], 0., np.inf)

    # test that the predicted value is as expected
    fe_usa_1990 = 1.
    re_usa_1990 = np.exp(.1+.2+.3)  # unchanged, since it is alpha_all that is now 1.
    assert_almost_equal(pred,
                        vars['p']['mu_age'].trace() * fe_usa_1990 * re_usa_1990)


    ### Prediction case 4: see that prediction of CAN includes region and super-region effect, but not USA effect
    pred = dismod_mr.model.covariates.predict_for(d, d.parameters['p'],
                                         'all', 'total', 'all',
                                         'CAN', 'male', 1990,
                                         0., vars['p'], 0., np.inf)

    # test that the predicted value is as expected
    fe = 1.
    re = np.exp(0.+.2+.3)  # unchanged, since it is alpha_all that is now 1.
    assert_almost_equal(pred,
                        vars['p']['mu_age'].trace() * fe * re)



    # create model and priors
    vars = dismod_mr.model.process.age_specific_rate(d, 'p', 'USA', 'male', 1990, None, None, None)

    # fit model
    m = mc.MCMC(vars)
    for n in m.stochastics:
        m.use_step_method(mc.NoStepper, n)
    m.sample(3)

    # check estimates
    pi_usa = dismod_mr.model.covariates.predict_for(d, d.parameters['p'],
                                         'USA', 'male', 1990,
                                         'USA', 'male', 1990,
                                         0., vars['p'], 0., np.inf)

    # test that the predicted value is as expected
    assert_almost_equal(pi_usa, vars['p']['mu_age'].trace())


    ### Prediction case 5: confirm that const RE prior with sigma = 0 does not crash

    d.parameters['p']['random_effects']['USA']['sigma'] = 0.
    d.parameters['p']['random_effects']['CAN']['sigma'] = 0.
        
    pred = dismod_mr.model.covariates.predict_for(d, d.parameters['p'],
                                       'all', 'total', 'all',
                                       'NAHI', 'male', 1990,
                                       0., vars['p'], 0., np.inf)




    d.vars = vars
    return d

# TODO: test predict for when there is a random effect (alpha)
# TODO: test predict when zerore=True
# TODO: test predicting for various values in the output template
def test_predict_for_wo_data():
    """ Approach to testing predict_for function:

    1. Create model with known mu_age, known covariate values, known effect coefficients
    2. Setup MCMC with NoStepper for all stochs
    3. Sample to generate trace with known values
    4. Predict for results, and confirm that they match expected values
    """
    
    
    d = dismod_mr.data.ModelData()
    d.hierarchy, d.output_template = data_simulation.small_output()


    # create model and priors
    vars = dismod_mr.model.process.age_specific_rate(d, 'p', 'all', 'total', 'all', None, None, None)

    # fit model
    m = mc.MCMC(vars)
    m.sample(1)


    ### Prediction case 1: constant zero random effects, zero fixed effect coefficients

    # check estimates with priors on random effects
    d.parameters['p']['random_effects'] = {}
    for node in ['USA', 'NAHI', 'super-region-1', 'all']:
        d.parameters['p']['random_effects'][node] = dict(dist='Constant', mu=0, sigma=1.e-9) # zero out REs to see if test passes
        
    pred = dismod_mr.model.covariates.predict_for(d, d.parameters['p'],
                                         'all', 'total', 'all',
                                         'USA', 'male', 1990,
                                         0., vars['p'], 0., np.inf)


    ### Prediction case 2: constant non-zero random effects, zero fixed effect coefficients
    # FIXME: this test was failing because PyMC is drawing from the prior of beta[0] even though I asked for NoStepper
                                                      
    # check estimates with priors on random effects
    for i, node in enumerate(['USA', 'NAHI', 'super-region-1']):
        d.parameters['p']['random_effects'][node]['mu'] = (i+1.)/10.
        
    pred = dismod_mr.model.covariates.predict_for(d, d.parameters['p'],
                                         'all', 'total', 'all',
                                         'USA', 'male', 1990,
                                         0., vars['p'], 0., np.inf)

    # test that the predicted value is as expected
    fe_usa_1990 = np.exp(.5*vars['p']['beta'][0].value) # beta[0] is drawn from prior, even though I set it to NoStepper, see FIXME above
    re_usa_1990 = np.exp(.1+.2+.3)
    assert_almost_equal(pred,
                        vars['p']['mu_age'].trace() * fe_usa_1990 * re_usa_1990)


def test_predict_for_wo_effects():
    """ Approach to testing predict_for function:

    1. Create model with known mu_age, known covariate values, known effect coefficients
    2. Setup MCMC with NoStepper for all stochs
    3. Sample to generate trace with known values
    4. Predict for results, and confirm that they match expected values
    """
    
    # generate simulated data
    n = 5
    sigma_true = .025
    a = np.arange(0, 100, 1)
    pi_age_true = .0001 * (a * (100. - a) + 100.)
    
    d = dismod_mr.data.ModelData()
    d.input_data = data_simulation.simulated_age_intervals('p', n, a, pi_age_true, sigma_true)
    d.hierarchy, d.output_template = data_simulation.small_output()


    # create model and priors
    vars = dismod_mr.model.process.age_specific_rate(d, 'p', 'NAHI', 'male', 2005, None, None, None, include_covariates=False)

    # fit model
    m = mc.MCMC(vars)
    for n in m.stochastics:
        m.use_step_method(mc.NoStepper, n)
    m.sample(10)


    ### Prediction case: prediction should match mu age
        
    pred = dismod_mr.model.covariates.predict_for(d, d.parameters['p'],
                                         'NAHI', 'male', 2005,
                                         'USA', 'male', 1990,
                                         0., vars['p'], 0., np.inf)

    assert_almost_equal(pred,
                        vars['p']['mu_age'].trace())


def test_predict_for_w_region_as_reference():
    """ Approach to testing predict_for function:

    1. Create model with known mu_age, known covariate values, known effect coefficients
    2. Setup MCMC with NoStepper for all stochs
    3. Sample to generate trace with known values
    4. Predict for results, and confirm that they match expected values
    """
    
    # generate simulated data
    n = 5
    sigma_true = .025
    a = np.arange(0, 100, 1)
    pi_age_true = .0001 * (a * (100. - a) + 100.)
    
    d = dismod_mr.data.ModelData()
    d.input_data = data_simulation.simulated_age_intervals('p', n, a, pi_age_true, sigma_true)
    d.hierarchy, d.output_template = data_simulation.small_output()


    # create model and priors
    vars = dismod_mr.model.process.age_specific_rate(d, 'p', 'NAHI', 'male', 2005, None, None, None)

    # fit model
    m = mc.MCMC(vars)
    for n in m.stochastics:
        m.use_step_method(mc.NoStepper, n)
    m.sample(10)


    ### Prediction case 1: constant zero random effects, zero fixed effect coefficients

    # check estimates with priors on random effects
    d.parameters['p']['random_effects'] = {}
    for node in ['USA', 'NAHI', 'super-region-1', 'all']:
        d.parameters['p']['random_effects'][node] = dict(dist='Constant', mu=0, sigma=1.e-9) # zero out REs to see if test passes
        
    pred = dismod_mr.model.covariates.predict_for(d, d.parameters['p'],
                                         'NAHI', 'male', 2005,
                                         'USA', 'male', 1990,
                                         0., vars['p'], 0., np.inf)

    # test that the predicted value is as expected
    fe_usa_1990 = np.exp(0.)
    re_usa_1990 = np.exp(0.)
    assert_almost_equal(pred,
                        vars['p']['mu_age'].trace() * fe_usa_1990 * re_usa_1990)


    ### Prediction case 2: constant non-zero random effects, zero fixed effect coefficients

    # check estimates with priors on random effects
    for i, node in enumerate(['USA', 'NAHI', 'super-region-1', 'all']):
        d.parameters['p']['random_effects'][node]['mu'] = (i+1.)/10.
        
    pred = dismod_mr.model.covariates.predict_for(d, d.parameters['p'],
                                         'NAHI', 'male', 2005,
                                         'USA', 'male', 1990,
                                         0., vars['p'], 0., np.inf)

    # test that the predicted value is as expected
    fe_usa_1990 = np.exp(0.)
    re_usa_1990 = np.exp(.1)
    assert_almost_equal(pred,
                        vars['p']['mu_age'].trace() * fe_usa_1990 * re_usa_1990)


    ### Prediction case 3: random effect not constant, zero fixed effect coefficients

    # set random seed to make randomness reproducible
    np.random.seed(12345)
    pred = dismod_mr.model.covariates.predict_for(d, d.parameters['p'],
                                         'NAHI', 'male', 2005,
                                         'CAN', 'male', 1990,
                                         0., vars['p'], 0., np.inf)

    # test that the predicted value is as expected
    np.random.seed(12345)
    fe = np.exp(0.)
    re = np.exp(mc.rnormal(0., vars['p']['sigma_alpha'][3].trace()**-2))
    assert_almost_equal(pred.mean(0),
                        (vars['p']['mu_age'].trace().T * fe * re).T.mean(0))


def assert_almost_equal(x, y):
    log_offset_diff = np.log(x + 1.e-4) - np.log(y + 1.e-4)
    assert np.all(log_offset_diff**2 <= 1.e-4), 'expected approximate equality, found means of:\n  %s\n  %s' % (x.mean(1), y.mean(1))
    

if __name__ == '__main__':
    import nose
    nose.runmodule()
