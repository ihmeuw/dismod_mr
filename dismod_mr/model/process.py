""" Dismod-MR model creation methods"""

import pylab as pl
import pymc as mc
import scipy.interpolate
import networkx as nx
import pandas

from dismod_mr import data
import rate_model
import age_pattern
import age_integrating_model
import covariate_model
import similarity_prior_model
import expert_prior_model
reload(expert_prior_model)
reload(similarity_prior_model)
reload(age_pattern)
reload(covariate_model)
reload(rate_model)

def age_specific_rate(model, data_type, reference_area='all', reference_sex='total', reference_year='all',
                      mu_age=None, mu_age_parent=None, sigma_age_parent=None, 
                      rate_type='neg_binom', lower_bound=None, interpolation_method='linear',
                      include_covariates=True, zero_re=False):
    # TODO: expose (and document) interface for alternative rate_type as well as other options,
    # record reference values in the model
    """ Generate PyMC objects for model of epidemological age-interval data

    :Parameters:
      - `model` : data.ModelData
      - `data_type` : str, one of 'i', 'r', 'f', 'p', or 'pf'
      - `reference_area, reference_sex, reference_year` : the node of the model to fit consistently
      - `mu_age` : pymc.Node, will be used as the age pattern, set to None if not needed
      - `mu_age_parent` : pymc.Node, will be used as the age pattern of the parent of the root area, set to None if not needed
      - `sigma_age_parent` : pymc.Node, will be used as the standard deviation of the age pattern, set to None if not needed
      - `rate_type` : str, optional. One of 'beta_binom', 'binom', 'log_normal_model', 'neg_binom', 'neg_binom_lower_bound_model', 'neg_binom_model', 'normal_model', 'offest_log_normal', or 'poisson'
      - `lower_bound` : 
      - `interpolation_method` : str, optional, one of 'linear', 'nearest', 'zero', 'slinear', 'quadratic, or 'cubic'
      - `include_covariates` : boolean
      - `zero_re` : boolean, change one stoch from each set of siblings in area hierarchy to a 'sum to zero' deterministic

    :Results:
      - Returns dict of PyMC objects, including 'pi', the covariate adjusted predicted values for each row of data

    """
    name = data_type
    import data
    result = data.ModelVars()
    
    if (mu_age_parent != None and pl.any(pl.isnan(mu_age_parent))) \
           or (sigma_age_parent != None and pl.any(pl.isnan(sigma_age_parent))):
        mu_age_parent = None
        sigma_age_parent = None
        print 'WARNING: nan found in parent mu/sigma.  Ignoring'

    ages = pl.array(model.parameters['ages'])
    data = model.get_data(data_type)
    if lower_bound:
        lb_data = model.get_data(lower_bound)
    parameters = model.parameters.get(data_type, {})
    area_hierarchy = model.hierarchy

    vars = dismod3.data.ModelVars()
    vars += dict(data=data)

    if 'parameter_age_mesh' in parameters:
        knots = pl.array(parameters['parameter_age_mesh'])
    else:
        knots = pl.arange(ages[0], ages[-1]+1, 5)

    smoothing_dict = {'No Prior':pl.inf, 'Slightly':.5, 'Moderately': .05, 'Very': .005}
    if 'smoothness' in parameters:
        smoothing = smoothing_dict[parameters['smoothness']['amount']]
    else:
        smoothing = 0.

    if mu_age == None:
        vars.update(
            age_pattern.age_pattern(name, ages=ages, knots=knots, smoothing=smoothing, interpolation_method=interpolation_method)
            )
    else:
        vars.update(dict(mu_age=mu_age, ages=ages))

    vars.update(expert_prior_model.level_constraints(name, parameters, vars['mu_age'], ages))
    vars.update(expert_prior_model.derivative_constraints(name, parameters, vars['mu_age'], ages))

    if mu_age_parent != None:
        # setup a hierarchical prior on the simliarity between the
        # consistent estimate here and (inconsistent) estimate for its
        # parent in the areas hierarchy
        #weight_dict = {'Unusable': 10., 'Slightly': 10., 'Moderately': 1., 'Very': .1}
        #weight = weight_dict[parameters['heterogeneity']]
        vars.update(
            similarity_prior_model.similar('parent_similarity_%s'%name, vars['mu_age'], mu_age_parent, sigma_age_parent, 0.)
            )

        # also use this as the initial value for the age pattern, if it is not already specified
        if mu_age == None:
            if isinstance(mu_age_parent, mc.Node):  # TODO: test this code
                initial_mu = mu_age_parent.value
            else:
                initial_mu = mu_age_parent
                
            for i, k_i in enumerate(knots):
                vars['gamma'][i].value = (pl.log(initial_mu[k_i-ages[0]])).clip(-12,6)

    age_weights = pl.ones_like(vars['mu_age'].value) # TODO: use age pattern appropriate to the rate type
    if len(data) > 0:
        vars.update(
            age_integrating_model.age_standardize_approx(name, age_weights, vars['mu_age'], data['age_start'], data['age_end'], ages)
            )

        # uncomment the following to effectively remove alleffects
        #if 'random_effects' in parameters:
        #    for i in range(5):
        #        effect = 'sigma_alpha_%s_%d' % (name, i)
        #        parameters['random_effects'][effect] = dict(dist='TruncatedNormal', mu=.0001, sigma=.00001, lower=.00009, upper=.00011)
        #if 'fixed_effects' in parameters:
        #    for effect in ['x_sex', 'x_LDI_id_Updated_7July2011']:
        #        parameters['fixed_effects'][effect] = dict(dist='normal', mu=.0001, sigma=.00001)

        if include_covariates:
            vars.update(
                covariate_model.mean_covariate_model(name, vars['mu_interval'], data, parameters, model, reference_area, reference_sex, reference_year, zero_re=zero_re)
                )
        else:
            vars.update({'pi': vars['mu_interval']})

        ## ensure that all data has uncertainty quantified appropriately
        # first replace all missing se from ci
        missing_se = pl.isnan(data['standard_error']) | (data['standard_error'] < 0)
        data['standard_error'][missing_se] = (data['upper_ci'][missing_se] - data['lower_ci'][missing_se]) / (2*1.96)

        # then replace all missing ess with se
        missing_ess = pl.isnan(data['effective_sample_size'])
        data['effective_sample_size'][missing_ess] = data['value'][missing_ess]*(1-data['value'][missing_ess])/data['standard_error'][missing_ess]**2

        if rate_type == 'neg_binom':

            # warn and drop data that doesn't have effective sample size quantified, or is is non-positive
            missing_ess = pl.isnan(data['effective_sample_size']) | (data['effective_sample_size'] < 0)
            if sum(missing_ess) > 0:
                print 'WARNING: %d rows of %s data has invalid quantification of uncertainty.' % (sum(missing_ess), name)
                data['effective_sample_size'][missing_ess] = 0.0

            # warn and change data where ess is unreasonably huge
            large_ess = data['effective_sample_size'] >= 1.e10
            if sum(large_ess) > 0:
                print 'WARNING: %d rows of %s data have effective sample size exceeding 10 billion.' % (sum(large_ess), name)
                data['effective_sample_size'][large_ess] = 1.e10


            if 'heterogeneity' in parameters:
                lower_dict = {'Slightly': 9., 'Moderately': 3., 'Very': 1.}
                lower = lower_dict[parameters['heterogeneity']]
            else:
                lower = 1.

            # special case, treat pf data as poisson
            if data_type == 'pf':
                lower = 1.e12
            
            vars.update(
                covariate_model.dispersion_covariate_model(name, data, lower, lower*9.)
                )

            vars.update(
                rate_model.neg_binom_model(name, vars['pi'], vars['delta'], data['value'], data['effective_sample_size'])
                )
        elif rate_type == 'log_normal':

            # warn and drop data that doesn't have effective sample size quantified
            missing = pl.isnan(data['standard_error']) | (data['standard_error'] < 0)
            if sum(missing) > 0:
                print 'WARNING: %d rows of %s data has no quantification of uncertainty.' % (sum(missing), name)
                data['standard_error'][missing] = 1.e6

            # TODO: allow options for alternative priors for sigma
            vars['sigma'] = mc.Uniform('sigma_%s'%name, lower=.0001, upper=1., value=.01)
            #vars['sigma'] = mc.Exponential('sigma_%s'%name, beta=100., value=.01)
            vars.update(
                rate_model.log_normal_model(name, vars['pi'], vars['sigma'], data['value'], data['standard_error'])
                )
        elif rate_type == 'normal':

            # warn and drop data that doesn't have standard error quantified
            missing = pl.isnan(data['standard_error']) | (data['standard_error'] < 0)
            if sum(missing) > 0:
                print 'WARNING: %d rows of %s data has no quantification of uncertainty.' % (sum(missing), name)
                data['standard_error'][missing] = 1.e6

            vars['sigma'] = mc.Uniform('sigma_%s'%name, lower=.0001, upper=.1, value=.01)
            vars.update(
                rate_model.normal_model(name, vars['pi'], vars['sigma'], data['value'], data['standard_error'])
                )
        elif rate_type == 'binom':
            missing_ess = pl.isnan(data['effective_sample_size']) | (data['effective_sample_size'] < 0)
            if sum(missing_ess) > 0:
                print 'WARNING: %d rows of %s data has invalid quantification of uncertainty.' % (sum(missing_ess), name)
                data['effective_sample_size'][missing_ess] = 0.0
            vars += rate_model.binom(name, vars['pi'], data['value'], data['effective_sample_size'])
        elif rate_type == 'beta_binom':
            vars += rate_model.beta_binom(name, vars['pi'], data['value'], data['effective_sample_size'])
        elif rate_type == 'poisson':
            missing_ess = pl.isnan(data['effective_sample_size']) | (data['effective_sample_size'] < 0)
            if sum(missing_ess) > 0:
                print 'WARNING: %d rows of %s data has invalid quantification of uncertainty.' % (sum(missing_ess), name)
                data['effective_sample_size'][missing_ess] = 0.0

            vars += rate_model.poisson(name, vars['pi'], data['value'], data['effective_sample_size'])
        elif rate_type == 'offset_log_normal':
            vars['sigma'] = mc.Uniform('sigma_%s'%name, lower=.0001, upper=10., value=.01)
            vars += rate_model.offset_log_normal(name, vars['pi'], vars['sigma'], data['value'], data['standard_error'])
        else:
            raise Exception, 'rate_model "%s" not implemented' % rate_type
    else:
        if include_covariates:
            vars.update(
                covariate_model.mean_covariate_model(name, [], data, parameters, model, reference_area, reference_sex, reference_year, zero_re=zero_re)
                )
    if include_covariates:
        vars.update(expert_prior_model.covariate_level_constraints(name, model, vars, ages))


    if lower_bound and len(lb_data) > 0:
        vars['lb'] = age_integrating_model.age_standardize_approx('lb_%s'%name, age_weights, vars['mu_age'], lb_data['age_start'], lb_data['age_end'], ages)

        if include_covariates:

            vars['lb'].update(
                covariate_model.mean_covariate_model('lb_%s'%name, vars['lb']['mu_interval'], lb_data, parameters, model, reference_area, reference_sex, reference_year, zero_re=zero_re)
                )
        else:
            vars['lb'].update({'pi': vars['lb']['mu_interval']})

        vars['lb'].update(
            covariate_model.dispersion_covariate_model('lb_%s'%name, lb_data, 1e12, 1e13)  # treat like poisson
            )

        ## ensure that all data has uncertainty quantified appropriately
        # first replace all missing se from ci
        missing_se = pl.isnan(lb_data['standard_error']) | (lb_data['standard_error'] <= 0)
        lb_data['standard_error'][missing_se] = (lb_data['upper_ci'][missing_se] - lb_data['lower_ci'][missing_se]) / (2*1.96)

        # then replace all missing ess with se
        missing_ess = pl.isnan(lb_data['effective_sample_size'])
        lb_data['effective_sample_size'][missing_ess] = lb_data['value'][missing_ess]*(1-lb_data['value'][missing_ess])/lb_data['standard_error'][missing_ess]**2

        # warn and drop lb_data that doesn't have effective sample size quantified
        missing_ess = pl.isnan(lb_data['effective_sample_size']) | (lb_data['effective_sample_size'] <= 0)
        if sum(missing_ess) > 0:
            print 'WARNING: %d rows of %s lower bound data has no quantification of uncertainty.' % (sum(missing_ess), name)
            lb_data['effective_sample_size'][missing_ess] = 1.0

        vars['lb'].update(
            rate_model.neg_binom_lower_bound_model('lb_%s'%name, vars['lb']['pi'], vars['lb']['delta'], lb_data['value'], lb_data['effective_sample_size'])
            )

    result[data_type] = vars
    return result
    
def consistent(model, reference_area='all', reference_sex='total', reference_year='all', priors={}, zero_re=True, rate_type='neg_binom'):
    """ Generate PyMC objects for consistent model of epidemological data
    
    :Parameters:
      - `model` : data.ModelData
      - `data_type` : str, one of 'i', 'r', 'f', 'p', or 'pf'
      - `root_area, root_sex, root_year` : the node of the model to
        fit consistently
      - `priors` : dictionary, with keys for data types for lists of
        priors on age patterns
      - `zero_re` : boolean, change one stoch from each set of
        siblings in area hierarchy to a 'sum to zero' deterministic
      - `rate_type` : str or dict, optional. One of 'beta_binom',
        'binom', 'log_normal_model', 'neg_binom',
        'neg_binom_lower_bound_model', 'neg_binom_model',
        'normal_model', 'offest_log_normal', or 'poisson', optionally
        as a dict, with keys i, r, f, p, m_with

    :Results:
      - Returns dict of dicts of PyMC objects, including 'i', 'p',
        'r', 'f', the covariate adjusted predicted values for each row
        of data
    
    .. note::
      - dict priors can contain keys (t, 'mu') and (t, 'sigma') to
        tell the consistent model about the priors on levels for the
        age-specific rate of type t (these are arrays for mean and
        standard deviation a priori for mu_age[t]
      - it can also contain dicts keyed by t alone to insert empirical
        priors on the fixed effects and random effects

    """
    # TODO: refactor the way priors are handled
    # current approach is much more complicated than necessary
    for t in 'i r pf p rr f'.split():
        if t in priors:
            model.parameters[t]['random_effects'].update(priors[t]['random_effects'])
            model.parameters[t]['fixed_effects'].update(priors[t]['fixed_effects'])

    # if rate_type is a string, make it into a dict
    if type(rate_type) == str:
        rate_type = dict(i=rate_type, r=rate_type, f=rate_type, p=rate_type, m_with=rate_type)

    rate = {}
    ages = model.parameters['ages']

    for t in 'irf':
        rate[t] = age_specific_rate(model, t, reference_area, reference_sex, reference_year,
                                    mu_age=None, mu_age_parent=priors.get((t, 'mu')), sigma_age_parent=priors.get((t, 'sigma')),
                                    zero_re=zero_re, rate_type=rate_type[t])[t] # age_specific_rate()[t] is to create proper nesting of dict

        # set initial values from data
        if t in priors:
            if isinstance(priors[t], mc.Node):
                initial = priors[t].value
            else:
                initial = pl.array(priors[t])
        else:
            initial = rate[t]['mu_age'].value.copy()
            df = model.get_data(t)
            if len(df.index) > 0:
                mean_data = df.groupby(['age_start', 'age_end']).mean().delevel()
                for i, row in mean_data.T.iteritems():
                    start = row['age_start'] - rate[t]['ages'][0]
                    end = row['age_end'] - rate[t]['ages'][0]
                    initial[start:end] = row['value']

        for i,k in enumerate(rate[t]['knots']):
            rate[t]['gamma'][i].value = pl.log(initial[k - rate[t]['ages'][0]]+1.e-9)

    m_all = .01*pl.ones(101)
    df = model.get_data('m_all')
    if len(df.index) == 0:
        print 'WARNING: all-cause mortality data not found, using m_all = .01'
    else:
        mean_mortality = df.groupby(['age_start', 'age_end']).mean().delevel()

        knots = []
        for i, row in mean_mortality.T.iteritems():
            knots.append(pl.clip((row['age_start'] + row['age_end'] + 1.) / 2., 0, 100))
            
            m_all[knots[-1]] = row['value']

        # extend knots as constant beyond endpoints
        knots = sorted(knots)
        m_all[0] = m_all[knots[0]]
        m_all[100] = m_all[knots[-1]]

        knots.insert(0, 0)
        knots.append(100)

        m_all = scipy.interpolate.interp1d(knots, m_all[knots], kind='linear')(pl.arange(101))
    m_all = m_all[ages]

    logit_C0 = mc.Uniform('logit_C0', -15, 15, value=-10.)


    # use Runge-Kutta 4 ODE solver
    import dismod_ode

    N = len(m_all)
    num_step = 10  # double until it works
    ages = pl.array(ages, dtype=float)
    fun = dismod_ode.ode_function(num_step, ages, m_all)

    @mc.deterministic
    def mu_age_p(logit_C0=logit_C0,
                 i=rate['i']['mu_age'],
                 r=rate['r']['mu_age'],
                 f=rate['f']['mu_age']):

        # for acute conditions, it is silly to use ODE solver to
        # derive prevalence, and it can be approximated with a simple
        # transformation of incidence
        if r.min() > 5.99:
            return i / (r + m_all + f)
        
        C0 = mc.invlogit(logit_C0)

        x = pl.hstack((i, r, f, 1-C0, C0))
        y = fun.forward(0, x)

        susceptible = y[:N]
        condition = y[N:]

        p = condition / (susceptible + condition)
        p[pl.isnan(p)] = 0.
        return p

    p = age_specific_rate(model, 'p',
                          reference_area, reference_sex, reference_year,
                          mu_age_p,
                          mu_age_parent=priors.get(('p', 'mu')),
                          sigma_age_parent=priors.get(('p', 'sigma')),
                          zero_re=zero_re, rate_type=rate_type)['p']

    @mc.deterministic
    def mu_age_pf(p=p['mu_age'], f=rate['f']['mu_age']):
        return p*f
    pf = age_specific_rate(model, 'pf',
                           reference_area, reference_sex, reference_year,
                           mu_age_pf,
                           mu_age_parent=priors.get(('pf', 'mu')),
                           sigma_age_parent=priors.get(('pf', 'sigma')),
                           lower_bound='csmr',
                           include_covariates=False,
                           zero_re=zero_re)['pf']

    @mc.deterministic
    def mu_age_m(pf=pf['mu_age'], m_all=m_all):
        return (m_all - pf).clip(1.e-6, 1.e6)
    rate['m'] = age_specific_rate(model, 'm_wo',
                                  reference_area, reference_sex, reference_year,
                                  mu_age_m,
                                  None, None,
                                  include_covariates=False,
                                  zero_re=zero_re)['m_wo']

    @mc.deterministic
    def mu_age_rr(m=rate['m']['mu_age'], f=rate['f']['mu_age']):
        return (m+f) / m
    rr = age_specific_rate(model, 'rr',
                           reference_area, reference_sex, reference_year,
                           mu_age_rr,
                           mu_age_parent=priors.get(('rr', 'mu')),
                           sigma_age_parent=priors.get(('rr', 'sigma')),
                           rate_type='log_normal',
                           include_covariates=False,
                           zero_re=zero_re)['rr']

    @mc.deterministic
    def mu_age_smr(m=rate['m']['mu_age'], f=rate['f']['mu_age'], m_all=m_all):
        return (m+f) / m_all
    smr = age_specific_rate(model, 'smr',
                            reference_area, reference_sex, reference_year,
                            mu_age_smr,
                            mu_age_parent=priors.get(('smr', 'mu')),
                            sigma_age_parent=priors.get(('smr', 'sigma')),
                            rate_type='log_normal',
                            include_covariates=False,
                            zero_re=zero_re)['smr']

    @mc.deterministic
    def mu_age_m_with(m=rate['m']['mu_age'], f=rate['f']['mu_age']):
        return m+f
    m_with = age_specific_rate(model, 'm_with',
                               reference_area, reference_sex, reference_year,
                               mu_age_m_with,
                               mu_age_parent=priors.get(('m_with', 'mu')),
                               sigma_age_parent=priors.get(('m_with', 'sigma')),
                               include_covariates=False,
                               zero_re=zero_re, rate_type=rate_type['m_with'])['m_with']
    
    # duration = E[time in bin C]
    @mc.deterministic
    def mu_age_X(r=rate['r']['mu_age'], m=rate['m']['mu_age'], f=rate['f']['mu_age']):
        hazard = r + m + f
        pr_not_exit = pl.exp(-hazard)
        X = pl.empty(len(hazard))
        X[-1] = 1 / hazard[-1]
        for i in reversed(range(len(X)-1)):
            X[i] = pr_not_exit[i] * (X[i+1] + 1) + 1 / hazard[i] * (1 - pr_not_exit[i]) - pr_not_exit[i]
        return X
    X = age_specific_rate(model, 'X',
                          reference_area, reference_sex, reference_year,
                          mu_age_X,
                          mu_age_parent=priors.get(('X', 'mu')),
                          sigma_age_parent=priors.get(('X', 'sigma')),
                          rate_type='normal',
                          include_covariates=True,
                          zero_re=zero_re)['X']

    vars = rate
    vars.update(logit_C0=logit_C0, p=p, pf=pf, rr=rr, smr=smr, m_with=m_with, X=X)
    return vars


# TODO: refactor emp_priors into a class and document them
def emp_priors(dm, reference_area, reference_sex, reference_year):
    import dismod3.utils
    param_type = dict(i='incidence', p='prevalence', r='remission', f='excess-mortality', rr='relative-risk', pf='prevalence_x_excess-mortality', m_with='mortality')
    emp_priors = {}
    for t in 'i r pf p rr f'.split():
        key = dismod3.utils.gbd_key_for(param_type[t], reference_area, reference_year, reference_sex)
        mu = dm.get_mcmc('emp_prior_mean', key)
        sigma = dm.get_mcmc('emp_prior_std', key)
        
        if len(mu) == 101 and len(sigma) == 101:
            emp_priors[t, 'mu'] = mu
            emp_priors[t, 'sigma'] = sigma

    return emp_priors

def effect_priors(model, type):
    """ Extract effect coefficients from model vars for rate type
    
    :Parameters:
      - `model` : data.ModelData
      - `type` : str, one of 'i', 'r', 'f', 'p', or 'pf'
    
    """
    vars = model.vars[type]
    prior_vals = {}
    
    prior_vals['new_alpha'] = {}
    if 'alpha' in vars:
        for n, col in zip(vars['alpha'], vars['U'].columns):
            if isinstance(n, mc.Node):
                stats = n.stats()
                if stats:
                    #prior_vals['new_alpha'][col] = dict(dist='TruncatedNormal', mu=stats['mean'], sigma=stats['standard deviation'], lower=-5., upper=5.)
                    prior_vals['new_alpha'][col] = dict(dist='Constant', mu=stats['mean'], sigma=stats['standard deviation'])

        # uncomment below to save empirical prior on sigma_alpha, the dispersion of the random effects
        for n in vars['sigma_alpha']:
            stats = n.stats()
            prior_vals['new_alpha'][n.__name__] = dict(dist='TruncatedNormal', mu=stats['mean'], sigma=stats['standard deviation'], lower=.01, upper=.5)

    prior_vals['new_beta'] = {}
    if 'beta' in vars:
        for n, col in zip(vars['beta'], vars['X'].columns):
            stats = n.stats()
            if stats:
                #prior_vals['new_beta'][col] = dict(dist='normal', mu=stats['mean'], sigma=stats['standard deviation'], lower=-pl.inf, upper=pl.inf)
                prior_vals['new_beta'][col] = dict(dist='Constant', mu=stats['mean'])

    return prior_vals
