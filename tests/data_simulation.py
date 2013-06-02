import numpy as np
import pymc as mc
import pandas
import networkx as nx

from dismod_mr import data

def simulated_age_intervals(data_type, n, a, pi_age_true, sigma_true):
    # choose age intervals to measure
    age_start = np.array(mc.runiform(0, 100, n), dtype=int)
    age_start.sort()  # sort to make it easy to discard the edges when testing
    age_end = np.array(mc.runiform(age_start+1, np.minimum(age_start+10,100)), dtype=int)

    # find truth for the integral across the age intervals
    import scipy.integrate
    pi_interval_true = [scipy.integrate.trapz(pi_age_true[a_0i:(a_1i+1)]) / (a_1i - a_0i) 
                        for a_0i, a_1i in zip(age_start, age_end)]

    # generate covariates that add explained variation
    X = mc.rnormal(0., 1.**2, size=(n,3))
    beta_true = [-.1, .1, .2]
    beta_true = [0, 0, 0]
    Y_true = np.dot(X, beta_true)

    # calculate the true value of the rate in each interval
    pi_true = pi_interval_true*np.exp(Y_true)

    # simulate the noisy measurement of the rate in each interval
    p = np.maximum(0., mc.rnormal(pi_true, 1./sigma_true**2.))

    # store the simulated data in a pandas DataFrame
    data = pandas.DataFrame(dict(value=p, age_start=age_start, age_end=age_end,
                                 x_0=X[:,0], x_1=X[:,1], x_2=X[:,2]))
    data['effective_sample_size'] = np.maximum(p*(1-p)/sigma_true**2, 1.)

    data['standard_error'] = np.nan
    data['upper_ci'] = np.nan
    data['lower_ci'] = np.nan

    data['year_start'] = 2005.  # TODO: make these vary
    data['year_end'] = 2005.
    data['sex'] = 'total'
    data['area'] = 'all'
    data['data_type'] = data_type
    
    return data

def small_output():

    # generate a moderately complicated hierarchy graph for the model
    hierarchy = nx.DiGraph()
    hierarchy.add_node('all')
    hierarchy.add_edge('all', 'super-region-1', weight=.1)
    hierarchy.add_edge('super-region-1', 'NAHI', weight=.1)
    hierarchy.add_edge('NAHI', 'CAN', weight=.1)
    hierarchy.add_edge('NAHI', 'USA', weight=.1)

    output_template=pandas.DataFrame(dict(year=[1990, 1990, 2005, 2005, 2010, 2010]*2,
                                          sex=['male', 'female']*3*2,
                                          x_0=[.5]*6*2,
                                          x_1=[0.]*6*2,
                                          x_2=[.5]*6*2,
                                          pop=[50.]*6*2,
                                          area=['CAN']*6 + ['USA']*6))

    return hierarchy, output_template

def simple_model(N):
    model = data.ModelData()
    model.input_data = pandas.DataFrame(index=range(N))
    initialize_input_data(model.input_data)

    return model


def initialize_input_data(input_data):
    input_data['age_start'] = 0
    input_data['age_end'] = 1
    input_data['year_start'] = 2005.
    input_data['year_end'] = 2005.
    input_data['sex'] = 'total'
    input_data['data_type'] = 'p'
    input_data['standard_error'] = np.nan
    input_data['upper_ci'] = np.nan
    input_data['lower_ci'] = np.nan
    input_data['area'] = 'all'


def add_quality_metrics(df):
    df['abs_err'] = df['true'] - df['mu_pred']
    df['rel_err'] = (df['true'] - df['mu_pred']) / df['true'].mean() # rel error normalized by crude mean of observed data
    df['covered?'] = (df['true'] >= df['lb_pred']) & (df['true'] <= df['ub_pred'])

def initialize_results(model):
    model.results = dict(param=[], bias=[], rel_bias=[], mare=[], mae=[], pc=[], time=[])

def finalize_results(model):
    model.results = pandas.DataFrame(model.results, columns='param bias rel_bias mae mare pc time'.split())

def add_to_results(model, name):
    df = getattr(model, name)
    model.results['param'].append(name)
    model.results['bias'].append(df['abs_err'].mean())
    model.results['rel_bias'].append(df['rel_err'].mean())
    model.results['mae'].append((np.median(np.absolute(df['abs_err'].dropna()))))
    model.results['mare'].append(np.median(np.absolute(df['rel_err'].dropna())))
    model.results['pc'].append(df['covered?'].mean())
    model.results['time'].append(model.mcmc.wall_time)


