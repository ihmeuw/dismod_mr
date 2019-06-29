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
""" Module for DisMod-MR model fitting methods"""

import sys, time
import numpy as np, pymc as mc, networkx as nx

def asr(model, data_type, iter=2000, burn=1000, thin=1, tune_interval=100, verbose=False):
    """ Fit data model for one epidemiologic parameter using MCMC

    :Parameters:
      - `model` : data.ModelData
      - `data_type` : str, one of 'i', 'r', 'f', 'p', or 'pf'
      - `iter` : int, number of posterior samples fit
      - `burn` : int, number of posterior samples to discard as burn-in
      - `thin` : int, samples thinned by this number
      - `tune_interval` : int
      - `verbose` : boolean

    :Results:
      - returns a pymc.MCMC object created from vars, that has been fit with MCMC

    .. note::
      - `burn` must be less than `iter`
      - `thin` must be less than `iter` minus `burn`

    """
    assert burn < iter, 'burn must be less than iter'
    assert thin < iter - burn, 'thin must be less than iter-burn'

    vars = model.vars[data_type]

    start_time = time.time()
    map = mc.MAP(vars)
    m = mc.MCMC(vars)

    ## use MAP to generate good initial conditions
    try:
        method='fmin_powell'
        tol=.001

        logger.info('finding initial values')
        find_asr_initial_vals(vars, method, tol, verbose)

        logger.info('\nfinding MAP estimate')
        map.fit(method=method, tol=tol, verbose=verbose)

        if verbose:
            print_mare(vars)
        logger.info('\nfinding step covariances estimate')
        setup_asr_step_methods(m, vars)

        logger.info('\nresetting initial values (1)')
        find_asr_initial_vals(vars, method, tol, verbose)
        logger.info('\nresetting initial values (2)\n')
        map.fit(method=method, tol=tol, verbose=verbose)
    except KeyboardInterrupt:
        logger.warning('Initial condition calculation interrupted')

    ## use MCMC to fit the model
    print_mare(vars)

    logger.info('sampling from posterior\n')
    m.iter=iter
    m.burn=burn
    m.thin=thin
    if verbose:
        try:
            m.sample(m.iter, m.burn, m.thin, tune_interval=tune_interval, progress_bar=True, progress_bar_fd=sys.stdout)
        except TypeError:
            m.sample(m.iter, m.burn, m.thin, tune_interval=tune_interval, progress_bar=False, verbose=verbose)
    else:
        m.sample(m.iter, m.burn, m.thin, tune_interval=tune_interval, progress_bar=False)

    m.wall_time = time.time() - start_time

    model.map = map
    model.mcmc = m

    return model.map, model.mcmc

def consistent(model, iter=2000, burn=1000, thin=1, tune_interval=100, verbose=False):
    """Fit data model for all epidemiologic parameters using MCMC

    :Parameters:
      - `model` : data.ModelData
      - `iter` : int, number of posterior samples fit
      - `burn` : int, number of posterior samples to discard as burn-in
      - `thin` : int, samples thinned by this number
      - `tune_interval` : int
      - `verbose` : boolean

    :Results:
      - returns a pymc.MCMC object created from vars, that has been fit with MCMC

    .. note::
      - `burn` must be less than `iter`
      - `thin` must be less than `iter` minus `burn`

    """
    assert burn < iter, 'burn must be less than iter'
    assert thin < iter - burn, 'thin must be less than iter-burn'

    param_types = 'i r f p pf rr smr m_with X'.split()

    vars = model.vars

    start_time = time.time()
    map = mc.MAP(vars)
    m = mc.MCMC(vars)

    ## use MAP to generate good initial conditions
    try:
        method='fmin_powell'
        tol=.001

        logger.info('fitting submodels')
        find_consistent_spline_initial_vals(vars, method, tol, verbose)

        for t in param_types:
            find_re_initial_vals(vars[t], method, tol, verbose)
            logger.heartbeat()

        find_consistent_spline_initial_vals(vars, method, tol, verbose)
        logger.heartbeat()

        for t in param_types:
            find_fe_initial_vals(vars[t], method, tol, verbose)
            logger.heartbeat()

        find_consistent_spline_initial_vals(vars, method, tol, verbose)
        logger.heartbeat()

        for t in param_types:
            find_dispersion_initial_vals(vars[t], method, tol, verbose)
            logger.heartbeat()

        logger.info('\nfitting all stochs\n')
        map.fit(method=method, tol=tol, verbose=verbose)

        if verbose:
            from fit_posterior import inspect_vars
            print(inspect_vars({}, vars))

    except KeyboardInterrupt:
        logger.warning('Initial condition calculation interrupted')

    ## use MCMC to fit the model

    try:
        logger.info('finding step covariances')
        vars_to_fit = [[vars[t].get('p_obs'), vars[t].get('pi_sim'), vars[t].get('smooth_gamma'), vars[t].get('parent_similarity'),
                        vars[t].get('mu_sim'), vars[t].get('mu_age_derivative_potential'), vars[t].get('covariate_constraint')] for t in param_types]
        max_knots = max([len(vars[t]['gamma']) for t in 'irf'])
        for i in range(max_knots):
            stoch = [vars[t]['gamma'][i] for t in 'ifr' if i < len(vars[t]['gamma'])]

            if verbose:
                print('finding Normal Approx for', [n.__name__ for n in stoch])
            try:
                na = mc.NormApprox(vars_to_fit + stoch)
                try:
                    na.fit(method='fmin_powell', verbose=verbose)
                except ZeroDivisionError as e:
                    print('Error that often happens with little data')
                    print(e)
                cov = np.array(np.linalg.inv(-na.hess), order='F')
                if np.all(np.linalg.eigvals(cov) >= 0):
                    m.use_step_method(mc.AdaptiveMetropolis, stoch, cov=cov)
                else:
                    raise ValueError
            except ValueError:
                if verbose:
                    print('cov matrix is not positive semi-definite')
                m.use_step_method(mc.AdaptiveMetropolis, stoch)

            logger.heartbeat()

        for t in param_types:
            setup_asr_step_methods(m, vars[t], vars_to_fit)

            # reset values to MAP
            find_consistent_spline_initial_vals(vars, method, tol, verbose)
            logger.heartbeat()
        map.fit(method=method, tol=tol, verbose=verbose)
        logger.heartbeat()
    except KeyboardInterrupt:
        logger.warning('Initial condition calculation interrupted')

    logger.info('\nsampling from posterior distribution\n')
    m.iter=iter
    m.burn=burn
    m.thin=thin
    if verbose:
        try:
            m.sample(m.iter, m.burn, m.thin, tune_interval=tune_interval, progress_bar=True, progress_bar_fd=sys.stdout)
        except TypeError:
            m.sample(m.iter, m.burn, m.thin, tune_interval=tune_interval, progress_bar=False, verbose=verbose)
    else:
        m.sample(m.iter, m.burn, m.thin, tune_interval=tune_interval, progress_bar=False)
    m.wall_time = time.time() - start_time

    model.map = map
    model.mcmc = m

    return model.map, model.mcmc

""" Routines for fitting disease models"""

## set number of threads to avoid overburdening cluster computers
try:
    import mkl
    mkl.set_num_threads(1)
except ImportError:
    pass
def print_mare(vars):
    if 'p_obs' in vars:
        are = np.atleast_1d(np.absolute((vars['p_obs'].value - vars['pi'].value)/vars['pi'].value))
        print('mare:', np.round_(np.median(are), 2))

class Log:
    def info(self, msg):
        print(msg, flush=True)
    def warning(self, msg):
        print(msg, flush=True)
    def heartbeat(self):
        print('.', end=' ', flush=True)
logger = Log()

param_types = 'i r f p pf rr smr m_with X'.split()

def find_consistent_spline_initial_vals(vars, method, tol, verbose):
    ## generate initial value by fitting knots sequentially
    vars_to_fit = [vars['logit_C0']]
    for t in param_types:
        vars_to_fit += [vars[t].get('covariate_constraint'),
                        vars[t].get('mu_age_derivative_potential'), vars[t].get('mu_sim'),
                        vars[t].get('p_obs'), vars[t].get('parent_similarity'), vars[t].get('smooth_gamma'),]
    max_knots = max([len(vars[t]['gamma']) for t in 'irf'])
    for i in [max_knots]: #range(1, max_knots+1):
        if verbose:
            print('fitting first %d knots of %d' % (i, max_knots))
        vars_to_fit += [vars[t]['gamma'][:i] for t in 'irf']
        mc.MAP(vars_to_fit).fit(method=method, tol=tol, verbose=verbose)

        if verbose:
            from fit_posterior import inspect_vars
            print(inspect_vars({}, vars)[-10:])
        else:
            logger.heartbeat()


def find_asr_initial_vals(vars, method, tol, verbose):
    for outer_reps in range(3):
        find_spline_initial_vals(vars, method, tol, verbose)
        find_re_initial_vals(vars, method, tol, verbose)
        find_spline_initial_vals(vars, method, tol, verbose)
        find_fe_initial_vals(vars, method, tol, verbose)
        find_spline_initial_vals(vars, method, tol, verbose)
        find_dispersion_initial_vals(vars, method, tol, verbose)
        logger.heartbeat()

def find_spline_initial_vals(vars, method, tol, verbose):
    ## generate initial value by fitting knots sequentially
    vars_to_fit = [vars.get('p_obs'), vars.get('pi_sim'), vars.get('smooth_gamma'), vars.get('parent_similarity'),
                   vars.get('mu_sim'), vars.get('mu_age_derivative_potential'), vars.get('covariate_constraint')]

    for i, n in enumerate(vars['gamma']):
        if verbose:
            print('fitting first %d knots of %d' % (i+1, len(vars['gamma'])))
        vars_to_fit.append(n)
        mc.MAP(vars_to_fit).fit(method=method, tol=tol, verbose=verbose)
        if verbose:
            print_mare(vars)

def find_re_initial_vals(vars, method, tol, verbose):
    if 'hierarchy' not in vars:
        return

    col_map = dict([[key, i] for i,key in enumerate(vars['U'].columns)])

    for reps in range(3):
        for p in nx.traversal.bfs_tree(vars['hierarchy'], 'all'):
            successors = vars['hierarchy'].successors(p)
            if successors:
                #print successors

                vars_to_fit = [vars.get('p_obs'), vars.get('pi_sim'), vars.get('smooth_gamma'), vars.get('parent_similarity'),
                               vars.get('mu_sim'), vars.get('mu_age_derivative_potential'), vars.get('covariate_constraint')]
                vars_to_fit += [vars.get('alpha_potentials')]

                re_vars = [vars['alpha'][col_map[n]] for n in list(successors) + [p] if n in vars['U']]
                vars_to_fit += re_vars
                if len(re_vars) > 0:
                    mc.MAP(vars_to_fit).fit(method=method, tol=tol, verbose=verbose)

                #print np.round_([re.value for re in re_vars if isinstance(re, mc.Node)], 2)
                #print_mare(vars)

    #print 'sigma_alpha'
    vars_to_fit = [vars.get('p_obs'), vars.get('pi_sim'), vars.get('smooth_gamma'), vars.get('parent_similarity'),
                   vars.get('mu_sim'), vars.get('mu_age_derivative_potential'), vars.get('covariate_constraint')]
    vars_to_fit += [vars.get('sigma_alpha')]
    mc.MAP(vars_to_fit).fit(method=method, tol=tol, verbose=verbose)
    #print np.round_([s.value for s in vars['sigma_alpha']])
    #print_mare(vars)


def find_fe_initial_vals(vars, method, tol, verbose):
    vars_to_fit = [vars.get('p_obs'), vars.get('pi_sim'), vars.get('smooth_gamma'), vars.get('parent_similarity'),
                   vars.get('mu_sim'), vars.get('mu_age_derivative_potential'), vars.get('covariate_constraint')]
    vars_to_fit += [vars.get('beta')]  # include fixed effects in sequential fit
    mc.MAP(vars_to_fit).fit(method=method, tol=tol, verbose=verbose)
    #print_mare(vars)

def find_dispersion_initial_vals(vars, method, tol, verbose):
    vars_to_fit = [vars.get('p_obs'), vars.get('pi_sim'), vars.get('smooth_gamma'), vars.get('parent_similarity'),
                   vars.get('mu_sim'), vars.get('mu_age_derivative_potential'), vars.get('covariate_constraint')]
    vars_to_fit += [vars.get('eta'), vars.get('zeta')]
    mc.MAP(vars_to_fit).fit(method=method, tol=tol, verbose=verbose)
    #print_mare(vars)


def setup_asr_step_methods(m, vars, additional_stochs=[]):
    # groups RE stochastics that are suspected of being dependent
    groups = []
    fe_group = [n for n in vars.get('beta', []) if isinstance(n, mc.Stochastic)]
    ap_group = [n for n in vars.get('gamma', []) if isinstance(n, mc.Stochastic)]
    groups += [[g_i, g_j] for g_i, g_j in zip(ap_group[1:], ap_group[:-1])] + [fe_group, ap_group, fe_group+ap_group]

    for a in vars.get('hierarchy', []):
        group = []

        col_map = dict([[key, i] for i,key in enumerate(vars['U'].columns)])

        if a in vars['U']:
            for b in nx.shortest_path(vars['hierarchy'], 'all', a):
                if b in vars['U']:
                    n = vars['alpha'][col_map[b]]
                    if isinstance(n, mc.Stochastic):
                        group.append(n)
        groups.append(group)
        #if len(group) > 0:
            #group += ap_group
            #groups.append(group)
            #group += fe_group
            #groups.append(group)

    for stoch in groups:
        if len(stoch) > 0 and np.all([isinstance(n, mc.Stochastic) for n in stoch]):
            # only step certain stochastics, for understanding convergence
            #if 'gamma_i' not in stoch[0].__name__:
            #    print 'no stepper for', stoch
            #    m.use_step_method(mc.NoStepper, stoch)
            #    continue

            #print 'finding Normal Approx for', [n.__name__ for n in stoch]
            if additional_stochs == []:
                vars_to_fit = [vars.get('p_obs'), vars.get('pi_sim'), vars.get('smooth_gamma'), vars.get('parent_similarity'),
                               vars.get('mu_sim'), vars.get('mu_age_derivative_potential'), vars.get('covariate_constraint')]
            else:
                vars_to_fit = additional_stochs

            try:
                raise ValueError
                na = mc.NormApprox(vars_to_fit + stoch)
                na.fit(method='fmin_powell', verbose=0)
                cov = np.array(np.inv(-na.hess), order='F')
                #print 'opt:', np.round_([n.value for n in stoch], 2)
                #print 'cov:\n', cov.round(4)
                if np.all(np.eigvals(cov) >= 0):
                    m.use_step_method(mc.AdaptiveMetropolis, stoch, cov=cov)
                else:
                    raise ValueError
            except ValueError:
                #print 'cov matrix is not positive semi-definite'
                m.use_step_method(mc.AdaptiveMetropolis, stoch)

