""" Module for DisMod-MR graphics"""

import pylab as pl
import pymc as mc
import pandas
import networkx as nx

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f0', '#ffff33']

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

def plot_data_bars(df, style='book', color='black', label=None, max=500):
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
    data_bars = zip(df['age_start'], df['age_end'], df['value'])

    if len(data_bars) > max:
        import random
        data_bars = random.sample(data_bars, max)

    # make lists of x and y points, faster than ploting each bar
    # individually
    x = []
    y = []
    for a_0i, a_1i, p_i in data_bars:
        x += [a_0i, a_1i, pl.nan]
        y += [p_i, p_i, pl.nan]

    if style=='book':
        pl.plot(x, y, 's-', mew=1, mec='w', ms=4, color=color, label=label)
    elif style=='talk':
        pl.plot(x, y, 's-', mew=1, mec='w', ms=0,
                alpha=1.0, color=colors[2], linewidth=15, label=label)
    else:
        raise Exception, 'Unrecognized style: %s' % style

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
                '95% HPD interval': pl.vstack((node.value, node.value)).T}

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
    pl.figure(figsize=fig_size)
    try:
        ages = vars['i']['ages']  # not all data models have an ages key, but incidence always does
    except KeyError:
        ages = vars[data_types[0]]['ages']
    for j, t in enumerate(data_types):
        pl.subplot(plot_config[0], plot_config[1], j+1)
        if with_data == 1: 
            plot_data_bars(model.input_data[model.input_data['data_type'] == t], color='grey', label='Data')
        if 'knots' in vars[t]:
            knots = vars[t]['knots']
        else:
            knots = range(101)
        try:
            pl.plot(ages, vars[t]['mu_age'].stats()['mean'], 'k-', linewidth=2, label='Posterior')
            if with_ui == 1:
                pl.plot(ages[knots], vars[t]['mu_age'].stats()['95% HPD interval'][knots,:][:,0], 'k--', label='95% HPD')
                pl.plot(ages[knots], vars[t]['mu_age'].stats()['95% HPD interval'][knots,:][:,1], 'k--')
        except (TypeError, AttributeError, KeyError):
            print 'Could not generate output statistics'
            if t in vars:
                pl.plot(ages, vars[t]['mu_age'].value, 'k-', linewidth=2)
        if t in posteriors:
            pl.plot(ages, posteriors[t], color='b', linewidth=1)
        if (t, 'mu') in emp_priors:
            mu = (emp_priors[t, 'mu']+1.e-9)[::5]
            s = (emp_priors[t, 'sigma']+1.e-9)[::5]
            pl.errorbar(ages[::5], mu,
                        yerr=[mu - pl.exp(pl.log(mu) - (s/mu+.1)),
                              pl.exp(pl.log(mu) + (s/mu+.1)) - mu],
                        color='grey', linewidth=1, capsize=0)

        pl.xlabel('Age (years)')
        pl.ylabel(ylab[j])
        pl.title(t)
    
def plot_one_ppc(model, t):
    """ plot data and posterior predictive check
    
    :Parameters:
      - `model` : data.ModelData
      - `t` : str, data type of 'i', 'r', 'f', 'p', 'rr', 'm', 'X', 'pf', 'csmr'
    
    """
    stats = model.vars[t]['p_pred'].stats()
    if stats == None:
        return

    pl.figure()
    pl.title(t)

    x = model.vars[t]['p_obs'].value.__array__()
    y = x - stats['quantiles'][50]
    yerr = [stats['quantiles'][50] - pl.atleast_2d(stats['95% HPD interval'])[:,0],
            pl.atleast_2d(stats['95% HPD interval'])[:,1] - stats['quantiles'][50]]
    pl.errorbar(x, y, yerr=yerr, fmt='ko', mec='w', capsize=0,
                label='Obs vs Residual (Obs - Pred)')

    pl.xlabel('Observation')
    pl.ylabel('Residual (observation-prediction)')

    pl.grid()
    l,r,b,t = pl.axis()
    pl.hlines([0], l, r)
    pl.axis([l, r, y.min()*1.1 - y.max()*.1, -y.min()*.1 + y.max()*1.1])

def plot_one_effects(model, data_type):
    """ Plot random effects and fixed effects.
    
    :Parameters:
      - `model` : data.ModelData
      - `data_types` : str, one of 'i', 'r', 'f', 'p', 'rr', 'pf'
      
    """
    vars = model.vars[data_type]
    hierarchy = model.hierarchy
    
    pl.figure(figsize=(22, 17))
    for i, (covariate, effect) in enumerate([['U', 'alpha'], ['X', 'beta']]):
        if covariate not in vars:
            continue
        
        cov_name = list(vars[covariate].columns)
        
        if isinstance(vars.get(effect), mc.Stochastic):
            pl.subplot(1, 2, i+1)
            pl.title('%s_%s' % (effect, data_type))

            stats = vars[effect].stats()
            if stats:
                if effect == 'alpha':
                    index = sorted(pl.arange(len(cov_name)),
                                   key=lambda i: str(cov_name[i] in hierarchy and nx.shortest_path(hierarchy, 'all', cov_name[i]) or cov_name[i]))
                elif effect == 'beta':
                    index = pl.arange(len(cov_name))

                x = pl.atleast_1d(stats['mean'])
                y = pl.arange(len(x))

                xerr = pl.array([x - pl.atleast_2d(stats['95% HPD interval'])[:,0],
                                 pl.atleast_2d(stats['95% HPD interval'])[:,1] - x])
                pl.errorbar(x[index], y[index], xerr=xerr[:, index], fmt='bs', mec='w')

                l,r,b,t = pl.axis()
                pl.vlines([0], b-.5, t+.5)
                pl.hlines(y, l, r, linestyle='dotted')
                pl.xticks([l, 0, r])
                pl.yticks([])
                for i in index:
                    spaces = cov_name[i] in hierarchy and len(nx.shortest_path(hierarchy, 'all', cov_name[i])) or 0
                    pl.text(l, y[i], '%s%s' % (' * '*spaces, cov_name[i]), va='center', ha='left')
                pl.axis([l, r, -.5, t+.5])
                
        if isinstance(vars.get(effect), list):
            pl.subplot(1, 2, i+1)
            pl.title('%s_%s' % (effect, data_type))
            index = sorted(pl.arange(len(cov_name)),
                           key=lambda i: str(cov_name[i] in hierarchy and nx.shortest_path(hierarchy, 'all', cov_name[i]) or cov_name[i]))

            for y, i in enumerate(index):
                n = vars[effect][i]
                if isinstance(n, mc.Stochastic) or isinstance(n, mc.Deterministic):
                    stats = n.stats()
                    if stats:
                        x = pl.atleast_1d(stats['mean'])

                        xerr = pl.array([x - pl.atleast_2d(stats['95% HPD interval'])[:,0],
                                         pl.atleast_2d(stats['95% HPD interval'])[:,1] - x])
                        pl.errorbar(x, y, xerr=xerr, fmt='bs', mec='w')

            l,r,b,t = pl.axis()
            pl.vlines([0], b-.5, t+.5)
            pl.hlines(y, l, r, linestyle='dotted')
            pl.xticks([l, 0, r])
            pl.yticks([])

            for y, i in enumerate(index):
                spaces = cov_name[i] in hierarchy and len(nx.shortest_path(hierarchy, 'all', cov_name[i])) or 0
                pl.text(l, y, '%s%s' % (' * '*spaces, cov_name[i]), va='center', ha='left')

            pl.axis([l, r, -.5, t+.5])
                

            if effect == 'alpha':
                effect_str = ''
                for sigma in vars['sigma_alpha']:
                    stats = sigma.stats()
                    if stats:
                        effect_str += '%s = %.3f\n' % (sigma.__name__, stats['mean'])
                    else:
                        effect_str += '%s = %.3f\n' % (sigma.__name__, sigma.value)
                pl.text(r, t, effect_str, va='top', ha='right')
            elif effect == 'beta':
                effect_str = ''
                if 'eta' in vars:
                    eta = vars['eta']
                    stats = eta.stats()
                    if stats:
                        effect_str += '%s = %.3f\n' % (eta.__name__, stats['mean'])
                    else:
                        effect_str += '%s = %.3f\n' % (eta.__name__, eta.value)
                pl.text(r, t, effect_str, va='top', ha='right')


def plot_hists(vars):
    """ Plot histograms for all stochs in a dict or dict of dicts
    
    :Parameters:
      - `vars` : data.ModelData.vars

    """
    def hist(trace):
        pl.hist(trace, histtype='stepfilled', normed=True)
        pl.yticks([])
        ticks, labels = pl.xticks()
        pl.xticks(ticks[1:6:2], fontsize=8)

    plot_viz_of_stochs(vars, hist)
    pl.subplots_adjust(0,.1,1,1,0,.2)


def plot_acorr(model):
    def acorr(trace):
        if len(trace) > 50:
            pl.acorr(trace, normed=True, detrend=pl.mlab.detrend_mean, maxlags=50)
        pl.xticks([])
        pl.yticks([])
        l,r,b,t = pl.axis()
        pl.axis([-10, r, -.1, 1.1])

    plot_viz_of_stochs(model.vars, acorr, (12,9))
    pl.subplots_adjust(0,0,1,1,0,0)


def plot_trace(model):
    def show_trace(trace):
        pl.plot(trace)
        pl.xticks([])

    plot_viz_of_stochs(model.vars, show_trace, (12,9))
    pl.subplots_adjust(.05,.01,.99,.99,.5,.5)

def plot_viz_of_stochs(vars, viz_func, figsize=(8,6)):
    """ Plot autocorrelation for all stochs in a dict or dict of dicts
    
    :Parameters:
      - `vars` : dictionary
      - `viz_func` : visualazation function such as ``acorr``, ``show_trace``, or ``hist``
      - `figsize` : tuple, size of figure
    
    """
    pl.figure(figsize=figsize)

    cells, stochs = tally_stochs(vars)

    # for each stoch, make an autocorrelation plot for each dimension
    rows = pl.floor(pl.sqrt(cells))
    cols = pl.ceil(cells/rows)

    tile = 1
    for s in sorted(stochs, key=lambda s: s.__name__):
        trace = s.trace()
        if len(trace.shape) == 1:
            trace = trace.reshape((len(trace), 1))
        for d in range(len(pl.atleast_1d(s.value))):
            pl.subplot(rows, cols, tile)
            viz_func(pl.atleast_2d(trace)[:, d])
            pl.title('\n\n%s[%d]'%(s.__name__, d), va='top', ha='center', fontsize=8)
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
            nodes = vars[k].values()
        else:
            nodes = [vars[k]]

        # handle lists of stochs
        for n in nodes:
            if isinstance(n, list):
                nodes += n

        for n in nodes:
            if isinstance(n, mc.Stochastic) and not n.observed:
                trace = n.trace()
                if len(trace) > 0:
                    stochs.append(n)
                    cells += len(pl.atleast_1d(n.value))
    return cells, stochs

