# Copyright 2008-2019 University of Washington
#
# This file is part of DisMod-MR.
#
# DisMod-MR is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DisMod-MR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with DisMod-MR.  If not, see <http://www.gnu.org/licenses/>.
""" Module for DisMod-MR graphics"""

import numpy as np, matplotlib.pyplot as plt
import pymc as pm
import pandas
import networkx as nx

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f0', '#ffff33']

def asr(model, t):
    """ plot age-standardized rate fit and model data

    :Parameters:
      - `model` : data.ModelData
      - `t` : str, data type to plot, e.g 'i', 'r', 'f', 'p', 'rr', or
        'pf'
    """
    vars = model.vars
    ages = np.array(model.parameters['ages'])

    data_bars(model.get_data(t), color='grey', label='%s data'%t)

    if 'knots' in vars[t]:
        knots = vars[t]['knots']
    else:
        knots = range(101)

    plt.plot(ages, vars[t]['mu_age'].stats()['mean'], 'k-', linewidth=2, label='Posterior')

    plt.plot(ages[knots], vars[t]['mu_age'].stats()['95% HPD interval'][knots,:][:,0], 'k--', label='95% HPD')
    plt.plot(ages[knots], vars[t]['mu_age'].stats()['95% HPD interval'][knots,:][:,1], 'k--')


def all_plots_for(model, t, ylab, emp_priors):
    """ plot results of a fit

    :Parameters:
      - `model` : data.ModelData
      - `data_types` : list of str, data types listed as strings, default = ['i', 'r', 'f', 'p', 'rr', 'pf']
      - `ylab` : list of str, list of y-axis labels corresponding to `data_types`
      - `emp_priors` : dictionary

    """
    plot_fit(model, data_types=[t], ylab=ylab, plot_config=(1,1), fig_size=(8,8))
    plot_one_ppc(model, t)
    plot_one_effects(model, t)
    plot_acorr(model.vars[t])
    plot_hists(model.vars)

def data_bars(df, style='book', color='black', label=None, max=500):
    """ Plot data bars

    :Parameters:
      - `df` : pandas.DataFrame with columns age_start, age_end, value
      - `style` : str, either book or talk
      - `color` : str, any matplotlib color
      - `label` : str, figure label
      - `max` : int, number of data points to display

    .. note::
      - The 'talk' style uses fewer colors, thicker line widths, and larger marker sizes.
      - If there are more than `max` data points, a random sample of `max` data points will be selected to show.

    """
    data_bars = list(zip(df['age_start'], df['age_end'], df['value']))

    if len(list(data_bars)) > max:
        import random
        data_bars = random.sample(data_bars, max)

    # make lists of x and y points, faster than ploting each bar
    # individually
    x = []
    y = []
    for a_0i, a_1i, p_i in data_bars:
        x += [a_0i, a_1i, np.nan]
        y += [p_i, p_i, np.nan]

    if style=='book':
        plt.plot(x, y, 's-', mew=1, mec='w', ms=4, color=color, label=label)
    elif style=='talk':
        plt.plot(x, y, 's-', mew=1, mec='w', ms=0,
                alpha=1.0, color=colors[2], linewidth=15, label=label)
    else:
        raise Exception('Unrecognized style: %s' % style)

def my_stats(node):
    """ Convenience function to generate a stats dict even if the pymc.Node has no trace

    :Parameters:
      - `node` : pymc.PyMCObjects.Deterministic

    :Results:
      - dictionary of statistics

    """
    try:
        return node.stats()
    except AttributeError:
        return {'mean': node.value,
                '95% HPD interval': np.vstack((node.value, node.value)).T}

def plot_fit(model, data_types=['i', 'r', 'f', 'p', 'rr', 'pf'], ylab=['PY','PY','PY','Percent (%)','','PY'], plot_config=(2,3),
             with_data=True, with_ui=True, emp_priors={}, posteriors={}, fig_size=(10,6)):
    """ plot results of a fit

    :Parameters:
      - `model` : data.ModelData
      - `data_types` : list of str, data types listed as strings, default = ['i', 'r', 'f', 'p', 'rr', 'pf']
      - `ylab` : list of str, list of y-axis labels corresponding to `data_types`
      - `plot_config` : tuple, subplot arrangement
      - `with_data` : boolean, plot with data type `t`, default = True
      - `with_ui` : boolean, plot with uncertainty interval, default = True
      - `emp_priors` : dictionary
      - `posteriors` :
      - `fig_size` : tuple, size of figure, default = (8,6)

    .. note::
      - `data_types` and `ylab` must be the same length
      - graphing options, such as ``pylab.subplots_adjust`` and ``pylab.legend()`` may be used to additionally modify graphics

    **Examples:**

    .. sourcecode:: python

        dismod3.graphics.plot_fit(model, ['i', 'p'], ['PY', '%'], (1,2), with_data=False, fig_size=(10,4))
        pylab.subplots_adjust(wspace=.3)

    .. figure:: graphics_plot_fit_multiple.png
        :align: center

    .. sourcecode:: python

        dismod3.graphics.plot_fit(model, ['i', 'p'], ['PY', '%'], (1,2), fig_size=(8,8))
        pylab.legend()

    .. figure:: graphics_plot_fit_single.png
        :align: center

    """
    assert len(data_types) == len(ylab), 'data_types and y-axis labels are not the same length'

    vars = model.vars
    plt.figure(figsize=fig_size)
    try:
        ages = vars['i']['ages']  # not all data models have an ages key, but incidence always does
    except KeyError:
        ages = vars[data_types[0]]['ages']
    for j, t in enumerate(data_types):
        plt.subplot(plot_config[0], plot_config[1], j+1)
        if with_data == 1:
            data_bars(model.input_data[model.input_data['data_type'] == t], color='grey', label='Data')
        if 'knots' in vars[t]:
            knots = vars[t]['knots']
        else:
            knots = range(101)
        try:
            plt.plot(ages, vars[t]['mu_age'].stats()['mean'], 'k-', linewidth=2, label='Posterior')
            if with_ui == 1:
                plt.plot(ages[knots], vars[t]['mu_age'].stats()['95% HPD interval'][knots,:][:,0], 'k--', label='95% HPD')
                plt.plot(ages[knots], vars[t]['mu_age'].stats()['95% HPD interval'][knots,:][:,1], 'k--')
        except (TypeError, AttributeError, KeyError):
            print('Could not generate output statistics')
            if t in vars:
                plt.plot(ages, vars[t]['mu_age'].value, 'k-', linewidth=2)
        if t in posteriors:
            plt.plot(ages, posteriors[t], color='b', linewidth=1)
        if (t, 'mu') in emp_priors:
            mu = (emp_priors[t, 'mu']+1.e-9)[::5]
            s = (emp_priors[t, 'sigma']+1.e-9)[::5]
            plt.errorbar(ages[::5], mu,
                        yerr=[mu - np.exp(np.log(mu) - (s/mu+.1)),
                              np.exp(np.log(mu) + (s/mu+.1)) - mu],
                        color='grey', linewidth=1, capsize=0)

        plt.xlabel('Age (years)')
        plt.ylabel(ylab[j])
        plt.title(t)

def plot_one_ppc(model, t):
    """ plot data and posterior predictive check

    :Parameters:
      - `model` : data.ModelData
      - `t` : str, data type of 'i', 'r', 'f', 'p', 'rr', 'm', 'X', 'pf', 'csmr'

    """
    stats = {}
    trace = model.vars[t]['p_pred'].trace()
    stats['quantiles'] = {50: np.median(trace, axis=0)}
    stats['95% HPD interval'] = pm.utils.hpd(trace, .05)
    if stats == None:
        return

    plt.figure()
    plt.title(t)

    x = model.vars[t]['p_obs'].value.__array__()
    y = x - stats['quantiles'][50]
    yerr = [stats['quantiles'][50] - stats['95% HPD interval'][0],
            stats['95% HPD interval'][1] - stats['quantiles'][50]]
    plt.errorbar(x, y, yerr=yerr, fmt='ko', mec='w', capsize=0,
                label='Obs vs Residual (Obs - Pred)')

    plt.xlabel('Observation')
    plt.ylabel('Residual (observation-prediction)')

    plt.grid()
    l,r,b,t = plt.axis()
    plt.hlines([0], l, r)
    plt.axis([l, r, y.min()*1.1 - y.max()*.1, -y.min()*.1 + y.max()*1.1])

def effects(model, data_type, figsize=(22, 17)):
    """ Plot random effects and fixed effects.

    :Parameters:
      - `model` : data.ModelData
      - `data_types` : str, one of 'i', 'r', 'f', 'p', 'rr', 'pf'

    """
    vars = model.vars[data_type]
    hierarchy = model.hierarchy

    if figsize:
        plt.figure(figsize=figsize)
    for i, (covariate, effect) in enumerate([['U', 'alpha'], ['X', 'beta']]):
        if covariate not in vars:
            continue

        cov_name = list(vars[covariate].columns)

        if isinstance(vars.get(effect), pm.Stochastic):
            plt.subplot(1, 2, i+1)
            plt.title('%s_%s' % (effect, data_type))

            trace = vars[effect].trace()
            if trace:
                if effect == 'alpha':
                    index = sorted(np.arange(len(cov_name)),
                                   key=lambda i: str(cov_name[i] in hierarchy and nx.shortest_path(hierarchy, 'all', cov_name[i]) or cov_name[i]))
                elif effect == 'beta':
                    index = np.arange(len(cov_name))

                x = np.atleast_1d(np.mean(trace))
                y = np.arange(len(x))
                hpd = pm.utils.hpd(trace, 0.05)

                xerr = np.array([x - np.atleast_2d(hpd)[:,0],
                                 np.atleast_2d(hpd)[:,1] - x])
                plt.errorbar(x[index], y[index], xerr=xerr[:, index], fmt='s', mec='w', color=colors[1])

                l,r,b,t = plt.axis()
                plt.vlines([0], b-.5, t+.5)
                #plt.hlines(y, l, r, linestyle='dotted')
                plt.xticks([l, l/2, 0, r/2, r])
                plt.yticks([])
                for i in index:
                    spaces = cov_name[i] in hierarchy and len(nx.shortest_path(hierarchy, 'all', cov_name[i])) or 0
                    plt.text(l, y[i], ' %s%s' % (' * '*spaces, cov_name[i]), va='center', ha='left')
                plt.axis([l, r, -.5, t+.5])

        if isinstance(vars.get(effect), list):
            plt.subplot(1, 2, i+1)
            plt.title('%s_%s' % (effect, data_type))
            index = sorted(np.arange(len(cov_name)),
                           key=lambda i: str(cov_name[i] in hierarchy and nx.shortest_path(hierarchy, 'all', cov_name[i]) or cov_name[i]))

            for y, i in enumerate(index):
                n = vars[effect][i]
                if isinstance(n, pm.Stochastic) or isinstance(n, pm.Deterministic):
                    trace = n.trace()
                    x = np.atleast_1d(np.mean(trace))
                    hpd = pm.utils.hpd(trace, 0.05)

                    xerr = np.array([x - np.atleast_2d(hpd)[:,0],
                                     np.atleast_2d(hpd)[:,1] - x])
                    plt.errorbar(x, y, xerr=xerr, fmt='s', mec='w', color=colors[1])

            l,r,b,t = plt.axis()
            plt.vlines([0], b-.5, t+.5)
            #plt.hlines(y, l, r, linestyle='dotted')
            plt.xticks([l, l/2, 0, r/2, r])
            plt.yticks([])

            for y, i in enumerate(index):
                spaces = cov_name[i] in hierarchy and len(nx.shortest_path(hierarchy, 'all', cov_name[i])) or 0
                plt.text(l, y, ' %s%s' % (' * '*spaces, cov_name[i]), va='center', ha='left')

            plt.axis([l, r, -.5, t+.5])


            effect_str = '\n'
            if effect == 'alpha':
                for sigma in vars['sigma_alpha']:
                    trace = sigma.trace()
                    if len(trace) > 0:
                        effect_str += '%s = %.3f -\n' % (sigma.__name__, np.mean(trace))  # FIXME: pylab doesn't align right correctly for lines that end in spaces
                    else:
                        effect_str += '%s = %.3f -\n' % (sigma.__name__, sigma.value)
                plt.text(r, t, effect_str, va='top', ha='right')
            elif effect == 'beta':
                if 'eta' in vars:
                    eta = vars['eta']
                    tr = eta.trace()
                    if len(tr) > 0:
                        effect_str += 'exp(%s) = %.3f -\n' % (eta.__name__, np.mean(np.exp(tr)))
                    else:
                        effect_str += 'exp(%s) = %.3f -\n' % (eta.__name__, np.exp(eta.value))
                plt.text(r, t, effect_str, va='top', ha='right')


def plot_hists(vars):
    """ Plot histograms for all stochs in a dict or dict of dicts

    :Parameters:
      - `vars` : data.ModelData.vars

    """
    def hist(trace):
        plt.hist(trace, histtype='stepfilled', normed=True)
        plt.yticks([])
        ticks, labels = plt.xticks()
        plt.xticks(ticks[1:6:2], fontsize=8)

    plot_viz_of_stochs(vars, hist)
    plt.subplots_adjust(0,.1,1,1,0,.2)


def plot_acorr(model):
    from matplotlib import mlab
    def acorr(trace):
        if len(trace) > 50:
            plt.acorr(trace, normed=True, detrend=mlab.detrend_mean, maxlags=50)
        plt.xticks([])
        plt.yticks([])
        l,r,b,t = plt.axis()
        plt.axis([-10, r, -.1, 1.1])

    plot_viz_of_stochs(model.vars, acorr, (12,9))
    plt.subplots_adjust(0,0,1,1,0,0)


def plot_trace(model):
    def show_trace(trace):
        plt.plot(trace)
        plt.xticks([])

    plot_viz_of_stochs(model.vars, show_trace, (12,9))
    plt.subplots_adjust(.05,.01,.99,.99,.5,.5)

def plot_viz_of_stochs(vars, viz_func, figsize=(8,6)):
    """ Plot autocorrelation for all stochs in a dict or dict of dicts

    :Parameters:
      - `vars` : dictionary
      - `viz_func` : visualazation function such as ``acorr``, ``show_trace``, or ``hist``
      - `figsize` : tuple, size of figure

    """
    plt.figure(figsize=figsize)

    cells, stochs = tally_stochs(vars)

    # for each stoch, make an autocorrelation plot for each dimension
    rows = np.floor(np.sqrt(cells))
    cols = np.ceil(cells/rows)

    tile = 1
    for s in sorted(stochs, key=lambda s: s.__name__):
        trace = s.trace()
        if len(trace.shape) == 1:
            trace = trace.reshape((len(trace), 1))
        for d in range(len(np.atleast_1d(s.value))):
            plt.subplot(rows, cols, tile)
            viz_func(np.atleast_2d(trace)[:, d])
            plt.title('\n\n%s[%d]'%(s.__name__, d), va='top', ha='center', fontsize=8)
            tile += 1


def tally_stochs(vars):
    """ Count number of stochastics in model

    :Parameters:
      - `vars` : dictionary

    """
    cells = 0
    stochs = []
    for k in vars.keys():
        # handle dicts and dicts of dicts by making a list of nodes
        if isinstance(vars[k], dict):
            nodes = list(vars[k].values())
        else:
            nodes = [vars[k]]

        # handle lists of stochs
        for n in nodes:
            if isinstance(n, list):
                nodes += n

        for n in nodes:
            if isinstance(n, pm.Stochastic) and not n.observed:
                trace = n.trace()
                if len(trace) > 0:
                    stochs.append(n)
                    cells += len(np.atleast_1d(n.value))
    return cells, stochs

def data_value_by_covariates(inp):
    """ Show raw values of data stratified by all x_* covariates

    :Parameters:
      - `inp` : pd.DataFrame of input data
    """
    X = inp.filter(like='x_')
    y = inp.value

    assert len(X.columns) <= 20, "FIXME: currently only works for up to 20 covariates"

    for i, c_i in enumerate(X.columns):
        plt.subplot(4,5,1+i)
        plt.title('\n'+c_i, va='top', fontsize=10)
        plt.ylabel('prevalence')
        plt.plot(X[c_i]+np.random.normal(size=len(y))*.03, y, 'k.', alpha=.5)
        plt.axis(xmin=-.1, xmax=1.1, ymin=-.1, ymax=1.1)
        plt.subplots_adjust(hspace=.5, wspace=.5)

def plot_residuals(dm):
    inp = dm.input_data
    plt.plot((inp.age_start+inp.age_end)/2 + 2*np.random.randn(len(inp.index)),
             inp.value - dm.vars['p']['mu_interval'].trace().mean(axis=0), 's',
             mec=colors[0], mew=1, color='w')
    plt.hlines([0],0,100, linestyle='dashed')
    plt.xlabel('Age (years)')
    plt.ylabel('Residual (observed $-$ predicted)')
    l,r,b,t = plt.axis()
    plt.vlines(dm.parameters['p']['parameter_age_mesh'],-1,1, linestyle='solid', color=colors[1])
    plt.axis([-5,105,b,t])
