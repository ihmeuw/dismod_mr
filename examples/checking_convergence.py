# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Checking convergence in DisMod-MR
# 
# This notebook provides some examples of running multiple chains and checking convergence in DisMod-MR.  Checking convergence is an important part of MCMC estimation.

# <codecell>

import numpy as np, pandas as pd, dismod_mr, pymc as pm, matplotlib.pyplot as plt, seaborn as sns
%matplotlib inline

# <codecell>

# set a random seed to ensure reproducible simulation results
np.random.seed(123456)

# <codecell>

# simulate data
n = 20

data = dict(age=np.random.randint(0, 10, size=n)*10,
            year=np.random.randint(1990, 2010, size=n))
data = pd.DataFrame(data)
data['value'] = (.1 + .001 * data.age) + np.random.normal(0., .01, size=n)

data['data_type'] = 'p'

data['age_start'] = data.age
data['age_end'] = data.age+10

# for prettier display, include jittered age near midpoint of age interval
data['jittered_age'] = .5*(data.age_start + data.age_end) + np.random.normal(size=n)

# keep things simple, no spatial random effects, no sex effect
data['area'] = 'all'
data['sex'] = 'total'

# quantification of uncertainty that says these numbers are believed to be quite precise
data['standard_error'] = -99
data['upper_ci'] = np.nan
data['lower_ci'] = np.nan
data['effective_sample_size'] = 1.e8


def new_model(data):
    # build the dismod_mr model
    dm = dismod_mr.data.ModelData()

    # set simple model parameters, for decent, fast computation
    dm.set_knots('p', [0,100])
    dm.set_level_bounds('p', lower=0, upper=1)
    dm.set_level_value('p', age_before=0, age_after=100, value=0)
    dm.set_heterogeneity('p', value='Slightly')
    dm.set_effect_prior('p', cov='x_sex', value=dict(dist='Constant', mu=0))
    
    # copy data into model 
    dm.input_data = data.copy()
    
    return dm

# <markdowncell>

# # Fit the model with too few iterations of MCMC

# <codecell>

dm1 = new_model(data)
dm1.setup_model('p', rate_model='neg_binom')
%time dm1.fit(how='mcmc', iter=10, burn=0, thin=1)
dm1.plot()

# <markdowncell>

# Fitting it again gives a different answer:

# <codecell>

dm2 = new_model(data)
dm2.setup_model('p', rate_model='neg_binom')
%time dm2.fit(how='mcmc', iter=10, burn=0, thin=1)
dm2.plot()

# <codecell>

dm1.vars['p']['gamma'][1].trace().mean()

# <codecell>

dm2.vars['p']['gamma'][1].trace().mean()

# <markdowncell>

# # Fit with more MCMC iterations

# <codecell>

dm1 = new_model(data)
dm1.setup_model('p', rate_model='neg_binom')
%time dm1.fit(how='mcmc', iter=10000, burn=5000, thin=5)

# <codecell>

dm2 = new_model(data)
dm2.setup_model('p', rate_model='neg_binom')
%time dm2.fit(how='mcmc', iter=10000, burn=5000, thin=5)

# <codecell>

dm1.vars['p']['gamma'][1].stats()

# <codecell>

dm2.vars['p']['gamma'][1].stats()

# <codecell>

dismod_mr.plot.plot_trace(dm1)

# <codecell>

dismod_mr.plot.plot_acorr(dm1)

# <markdowncell>

# # Running multiple chains
# 
# It is simple to run multiple chains sequentially in DisMod-MR, although I worry that this gives a false sense of security about the convergence.

# <codecell>

# setup a model and run the chain once

dm = new_model(data)
dm.setup_model('p', rate_model='neg_binom')
%time dm.fit(how='mcmc', iter=2000, burn=1000, thin=1)

# <codecell>

# to run it more times, use the sample method of the dm.mcmc object
# use the same iter/burn/thin settings for future convenience

for i in range(4):
    dm.mcmc.sample(iter=2000, burn=1000, thin=1)

# <codecell>

# calculate Gelman-Rubin statistic for all model variables
R_hat = pm.gelman_rubin(dm.mcmc)

# examine for gamma_p_100
R_hat['gamma_p_100']

# <codecell>


