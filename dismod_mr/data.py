""" Data Handling Class for DisMod-MR"""

import pandas as pd
import networkx as nx
import pymc as mc
import pylab as pl
import simplejson as json

import plot

def describe_vars(d):
    m = mc.Model(d)

    df = pd.DataFrame(columns=['type', 'value', 'logp'],
                          index=[n.__name__ for n in m.nodes],
                          dtype=object)
    for n in m.nodes:
        k = n.__name__
        df.ix[k, 'type'] = type(n).__name__

        if hasattr(n, 'value'):
            rav = pl.ravel(n.value)
            if len(rav) == 1:
                df.ix[k, 'value'] = n.value
            elif len(rav) > 1:
                df.ix[k, 'value'] = '%.1f, ...' % rav[0]

        df.ix[k, 'logp'] = getattr(n, 'logp', pl.nan)

    return df.sort('logp')


def check_convergence(vars):
    """ Apply a simple test of convergence to the model: compare
    autocorrelation at 25 lags to zero lags.  warn about convergence if it exceeds
    10% for any stoch """
    import dismod3
    cells, stochs = dismod_mr.plot.tally_stochs(vars)

    for s in sorted(stochs, key=lambda s: s.__name__):
        tr = s.trace()
        if len(tr.shape) == 1:
            tr = tr.reshape((len(tr), 1))
        for d in range(len(pl.atleast_1d(s.value))):
            for k in range(50,100):
                acorr = pl.dot(tr[:-k,d]-tr[:k,d].mean(), tr[k:,d]-tr[k:,d].mean()) / pl.dot(tr[k:,d]-tr[k:,d].mean(), tr[k:,d]-tr[k:,d].mean())
                if abs(acorr) > .5:
                    print 'potential non-convergence', s, acorr
                    return False
            
    return True

class ModelVars(dict):
    """ Container class for PyMC Node objects that make up the model

    Requirements:
    * access vars like a dictionary
    * add new vars with += (that functions like an update)
    ** pretty print information about what was added
    * .describe() the state of the nodes in the model
    ** say if the model has been run, and if it appears to have converged
    * .display() the model values in some informative graphical form (may need to be several functions)
    """
    def __iadd__(self, d):
        """ Over-ride += operator so that it updates dict with another
        dict, with verbose information about what is being added
        """
        #df = describe_vars(d)
        #print "Adding Variables:"
        #print df[:10]
        #if len(df.index) > 10:
        #    print '...\n(%d rows total)' % len(df.index)

        self.update(d)
        return self

    def __str__(self):
        return '%s\nkeys: %s' % (describe_vars(self), ', '.join(self.keys()))

    def describe(self):
        print describe_vars(self)

    def empirical_priors_from_fit(self, type_list=['i', 'r', 'f', 'p', 'rr']):
        """ Find empirical priors for asr of type t
        Parameters
        ----------
        type_list : list containing some of the folloring ['i', 'r', 'f', 'p', 'rr', 'pf', 'csmr', 'X']

        Results
        -------
        prior_dict, with distribution for each stoch in model
        """

        prior_dict = {}

        for t in type_list:
            if t in self:
                # TODO: eliminate unnecessary dichotomy in storing fe and re priors separately
                pdt = dict(random_effects={}, fixed_effects={})

                if 'U' in self[t]:
                    for i, re in enumerate(self[t]['U'].columns):
                        if isinstance(self[t]['alpha'][i], mc.Node):
                            pdt['random_effects'][re] = dict(dist='Constant', mu=self[t]['alpha'][i].stats()['mean'])
                        else:
                            pdt['random_effects'][re] = dict(dist='Constant', mu=self[t]['alpha'][i])

                if 'X' in self[t]:
                    for i, fe in enumerate(self[t]['X'].columns):
                        if isinstance(self[t]['beta'][i], mc.Node):
                            pdt['fixed_effects'][fe] = dict(dist='Constant', mu=self[t]['beta'][i].stats()['mean'])
                        else:
                            pdt['fixed_effects'][fe] = dict(dist='Constant', mu=self[t]['beta'][i])

                prior_dict[t] = pdt
        return prior_dict

class ModelData:
    """ ModelData object contains all information for a disease model:
        Data, model parameters, information about output
    """

    def __init__(self):
        self.input_data = pd.DataFrame(columns=('data_type value area sex age_start age_end year_start year_end' +
                                           ' standard_error effective_sample_size lower_ci upper_ci age_weights').split())
        self.output_template = pd.DataFrame(columns='data_type area sex year pop'.split())
        self.parameters = dict(i={}, p={}, r={}, f={}, rr={}, X={}, pf={}, ages=range(101))

        self.hierarchy = nx.DiGraph()
        self.hierarchy.add_node('all')

        self.nodes_to_fit = self.hierarchy.nodes()

        self.vars = ModelVars()

        self.model_settings = {} # TODO: determine if this is a good idea, think carefully about name

    def get_data(self, data_type):
        """ Select data of one type.
        
        :Parameters:
          - `data_type` : str, one of 'i', 'r', 'f', 'p', 'rr', 'pf', 'm', 'X', or 'csmr'
        
        :Results: 
          - DataFrame of selected data type.

        """
        if len(self.input_data) > 0:
            return self.input_data[self.input_data['data_type'] == data_type]
        else:
            return self.input_data

    def describe(self, data_type):
        G = self.hierarchy
        df = self.get_data(data_type)

        for n in nx.dfs_postorder_nodes(G, 'all'):
            G.node[n]['cnt'] = len(df[df['area']==n].index) + pl.sum([G.node[c]['cnt'] for c in G.successors(n)])
            G.node[n]['depth'] = nx.shortest_path_length(G, 'all', n)
        
        for n in nx.dfs_preorder_nodes(G, 'all'):
            if G.node[n]['cnt'] > 0:
                print ' *'*G.node[n]['depth'], n, int(G.node[n]['cnt'])

    def plot(self):
        import matplotlib.pyplot as plt, numpy as np
        import plot

        #for i, t in enumerate(self.input_data.data_type.unique()):

        t = 'p'
        a0 = self.parameters['ages'][0]
    
        plt.figure()
        plot.data_bars(self.get_data('p'), color=plot.colors[1])
    
        if t in self.vars:
            x = np.array(self.parameters['ages'])
            knots = self.vars[t]['knots']
            
            if not hasattr(self.vars[t]['mu_age'], 'trace'):
                pt = self.vars[t]['mu_age'].value
                plt.plot(x, pt, linewidth=3, color=plot.colors[0])
                plt.plot(knots, pt[knots-a0], 's', ms=15, mec='w', color=plot.colors[0])
            else:
            
                pred = self.vars['p']['mu_age'].trace()
                import pymc as mc
                ui =mc.utils.hpd(pred, .05)

                plt.plot(x[knots-a0], ui[knots-a0, :], '--', linewidth=2, color=plot.colors[0], alpha=1)

                plt.plot(x, pred.T, linewidth=10, color=plot.colors[0], alpha=.005)
                plt.plot(self.parameters['ages'], pred.mean(0), linewidth=5, color='w')

                xx = []
                yy = []
                for k_i in self.vars[t]['knots']:
                    xx += [k_i, k_i, np.nan]
                    yy += [0., pred.mean(axis=0)[k_i-a0], np.nan]
                plt.plot(xx, yy, linewidth=5, color='w')
                plt.plot(xx, yy, linewidth=3, color=plot.colors[0])
                plt.plot(self.parameters['ages'], pred.mean(0), linewidth=3, color=plot.colors[0])
                plt.plot(knots, pred.mean(axis=0)[knots-a0], 's', ms=15, mec='w', color=plot.colors[0])


                import pymc as mc
                ui =mc.utils.hpd(pred, .05)
                plt.plot(x, pred.mean(0), linewidth=5, color='w')
                plt.plot(x, pred.mean(0), linewidth=3, color=plot.colors[0])

        plt.axis(xmin=-5, xmax=105)

    def keep(self, areas=['all'], sexes=['male', 'female', 'total'], start_year=-pl.inf, end_year=pl.inf):
        """ Modify model to feature only desired area/sex/year(s)

        :Parameters:
          - `areas` : list of str, optional
          - `sexes` : list of str, optional
          - `start_year` : int, optional
          - `end_year` : int, optional

        """
        if 'all' not in areas:
            self.hierarchy.remove_node('all')
            for area in areas:
                self.hierarchy.add_edge('all', area)
            self.hierarchy = nx.bfs_tree(self.hierarchy, 'all')

            def relevant_row(i):
                area = self.input_data['area'][i]
                return (area in self.hierarchy) or (area == 'all')

            self.input_data = self.input_data.select(relevant_row)
            self.nodes_to_fit = set(self.hierarchy.nodes()) & set(self.nodes_to_fit)

        self.input_data = self.input_data.select(lambda i: self.input_data['sex'][i] in sexes)

        self.input_data = self.input_data.select(lambda i: self.input_data['year_end'][i] >= start_year)
        self.input_data = self.input_data.select(lambda i: self.input_data['year_start'][i] <= end_year)

        print 'kept %d rows of data' % len(self.input_data.index)

    def set_smoothness(self, rate_type, value):
        """ Set smoothness parameter for age-specific rate function of one type.
        
        :Parameters:
          - `rate_type` : str, one of 'i', 'r', 'f', 'p', 'rr', 'pf', 'm', 'X', or 'csmr'
          - `value` : str, one of 'No Prior', 'Slightly', 'Moderately', or 'Very',
            or non-negative float
        
        :Results: 
          - Changes the smoothing parameter in self.parameters

        """
        self.parameters[rate_type]['smoothness'] = dict(age_start=0, age_end=100, amount=value)

    def set_knots(self, rate_type, value):
        """ Set knots for age-specific rate function of one type.
        
        :Parameters:
          - `rate_type` : str, one of 'i', 'r', 'f', 'p', 'rr', 'pf',
            'm', 'X', or 'csmr'
          - `value` : list, positions knots, start and end must
            correspond to parameters['ages']
        
        :Results: 
          - Changes the knots in self.parameters[rate_type]

        """
        self.parameters[rate_type]['parameter_age_mesh'] = value

    def set_level_value(self, rate_type, age_before=None, age_after=None, value=0):
        """ Set level value for age-specific rate function of one
        type.
        
        :Parameters:
          - `rate_type` : str, one of 'i', 'r', 'f', 'p', 'rr', 'pf',
            'm', 'X', or 'csmr'
          - `age_before` : int, level value is applied for all ages
            less than this
          - `age_after` : int, level value is applied for all ages
            more than this
          - `value` : float, value of the age-specific rate function
            before and after specified ages
        
        :Results: 
          - Changes level_value in self.parameters[rate_type]

        """
        if age_before == None:
            age_before = 0
        if age_after == None:
            age_after = 0

        self.parameters[rate_type]['level_value'] = dict(age_before=age_before, age_after=age_after, value=value)

    def set_level_bounds(self, rate_type, lower=0, upper=1):
        """ Set level bounds for age-specific rate function of one
        type.
        
        :Parameters:
          - `rate_type` : str, one of 'i', 'r', 'f', 'p', 'rr', 'pf',
            'm', 'X', or 'csmr'
          - `lower` : float, minimum value of the age-specific rate
            function
          - `upper` : float, maximum value of the age-specific rate
            function

        
        :Results: 
          - Changes level_bounds in self.parameters[rate_type]

        """
        self.parameters[rate_type]['level_bounds'] = dict(lower=lower, upper=upper)

    def set_increasing(self, rate_type, age_start, age_end):
        """ Set increasing prior for age-specific rate function of one
        type.
        
        :Parameters:
          - `rate_type` : str, one of 'i', 'r', 'f', 'p', 'rr', 'pf',
            'm', 'X', or 'csmr'
          - `age_start` : int, minimum age of the age-specific rate
            function prior to increase
          - `age_end` : int, maximum age of the age-specific rate
            function prior to increase

        
        :Results: 
          - Changes increasing in self.parameters[rate_type]

        """
        self.parameters[rate_type]['increasing'] = dict(age_start=age_start, age_end=age_end)

    def set_decreasing(self, rate_type, age_start, age_end):
        """ Set decreasing prior for age-specific rate function of one
        type.
        
        :Parameters:
          - `rate_type` : str, one of 'i', 'r', 'f', 'p', 'rr', 'pf',
            'm', 'X', or 'csmr'
          - `age_start` : int, minimum age of the age-specific rate
            function prior to decrease
          - `age_end` : int, maximum age of the age-specific rate
            function prior to decrease

        
        :Results: 
          - Changes decreasing in self.parameters[rate_type]

        """
        self.parameters[rate_type]['decreasing'] = dict(age_start=age_start, age_end=age_end)

    def set_heterogeneity(self, rate_type, value):
        """ Set heterogeneity prior for age-specific rate function of one
        type.
        
        :Parameters:
          - `rate_type` : str, one of 'i', 'r', 'f', 'p'
          - `value` : str, one of 'Unusable', 'Very', 'Moderately', or
            'Slightly'

        
        :Results: 
          - Changes heterogeneity in self.parameters[rate_type]

        """
        self.parameters[rate_type]['heterogeneity'] = value


    def setup_model(self, rate_type=None, rate_model='neg_binom', include_covariates=True):
        """ Setup PyMC model vars based on current parameters and data
        :Parameters:
          - `rate_type` : str, optional
            if rate_type is provided, the model will be an age
            standardized rate model for the specified rate_type.
            otherwise, it will be a consistent model for all rate
            types.
          - `rate_model` : str, optional, one of 'beta_binom',
            'binom', 'log_normal', 'neg_binom',
            'neg_binom_lower_bound', 'neg_binom',
            'normal', 'offest_log_normal', or 'poisson'
            if rate_type is provided, this option specifies the
            rate model for data of this rate type.
          - `include_covariates` : bool, optional
            if rate_type is provided, this option specifies if the
            model for the rate type should include additional fixed
            and random effects
        :Notes:
        This method also creates methods fit and predict_for for the current object
        """

        if rate_type:
            import model
            self.vars = model.asr(self, rate_type,
                                  rate_type=rate_model,  # TODO: rename parameter in model.process.asr so this is less confusing
                                  include_covariates=include_covariates)

            self.model_settings['rate_type'] = rate_type

            def predict_for(area, sex, year):
                import model
                return model.covariates.predict_for(
                    self, self.parameters[rate_type],
                    'all', 'total', 'all',
                    area, sex, year,
                    1., self.vars[rate_type], 
                    self.parameters[rate_type]['level_bounds']['lower'], 
                    self.parameters[rate_type]['level_bounds']['upper'])

                
        else:
            assert 0
            self.vars = {}

        self.predict_for = predict_for

    def fit(self, how='mcmc', iter=10000, burn=5000, thin=5):
        """ Fit the model
        :Parameters:
          - `how` : str, 'mcmc' or 'map'
          'mcmc' is slower but provides uncertainty estimates
          - iter, burn, thin : int
          mcmc options

        :Notes:
        This must come after a call to .setup_model.
        """
        if 'rate_type' in self.model_settings:
            rate_type=self.model_settings['rate_type']

            import fit
            if how=='mcmc':
                self.map, self.mcmc = fit.asr(
                    self, rate_type,
                    iter=iter, burn=burn, thin=thin)
            elif how=='map':
                self.map = mc.MAP(self.vars[rate_type])
                fit.find_asr_initial_vals(
                    self.vars[rate_type], 'fmin_powell', tol=1e-3, verbose=0)
                self.map.fit(method='fmin_powell')
        elif 'consistent' in model_settings:
            assert 0, 'Not Yet Implemented'
        else:
            raise NotImplementedError, 'Need to call .setup_model before calling fit.'

    def predict_for(area, sex, year):
        """
        Predict
        """
        assert 0, 'Not yet implemented'
        import covariate_model
        reload(covariate_model)
        self.estimates = self.estimates.append(pd.DataFrame())

    def save(self, path):
        """ Saves all model data in human-readable files

        :Parameters:
          - `path` : str, directory to save in

        :Results:
          - Saves files to specified path, overwritting what was there before
        
        """

        self.input_data.to_csv(path + '/input_data.csv')
        self.output_template.to_csv(path + '/output_template.csv')
        json.dump(self.parameters, open(path + '/parameters.json', 'w'), indent=2)
        json.dump(dict(nodes=[[n, self.hierarchy.node[n]] for n in sorted(self.hierarchy.nodes())],
                       edges=[[u, v, self.hierarchy.edge[u][v]] for u,v in sorted(self.hierarchy.edges())]),
                  open(path + '/hierarchy.json', 'w'), indent=2)
        json.dump(list(self.nodes_to_fit), open(path + '/nodes_to_fit.json', 'w'), indent=2)

    @staticmethod
    def load(path):
        """ Load all model data
        
        :Parameters:
          - `path` : str, directory to save in
          
        :Results:
          - ModelData with all input data
          
        .. note::
          `path` must contain the following files 
            - :ref:`input_data-label` 
            - :ref:`output_template-label` 
            - :ref:`hierarchy-label`
            - :ref:`parameters-label`
            - :ref:`nodes_to_fit-label`
        
        """
        d = ModelData()

        # TODO: catch _csv.Error and retry, to give j drive time to sync
        d.input_data = pd.read_csv(path + '/input_data.csv')

        # ensure that certain columns are float
        #for field in 'value standard_error upper_ci lower_ci effective_sample_size'.split():
        #    #d.input_data.dtypes[field] = float  # TODO: figure out classy way like this, that works
        #    d.input_data[field] = pl.array(d.input_data[field], dtype=float)

        
        d.output_template = pd.DataFrame.from_csv(path + '/output_template.csv')
        
        d.parameters = json.load(open(path + '/parameters.json'))

        hierarchy = json.load(open(path + '/hierarchy.json'))
        d.hierarchy.add_nodes_from(hierarchy['nodes'])
        d.hierarchy.add_edges_from(hierarchy['edges'])

        d.nodes_to_fit = json.load(open(path + '/nodes_to_fit.json'))

        return d


load = ModelData.load

