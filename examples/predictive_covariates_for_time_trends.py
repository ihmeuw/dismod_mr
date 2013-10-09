# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Using predictive covariates to estimate time trends with DisMod-MR
# 
# The goal of this document is to demonstrate how DisMod-MR can produce estimates of time trends using predictive covariates.
# 
# To demonstrate this clearly, I have used simulated data: the true age pattern is linear, and prevalence levels vary over time in a way completely explained by covariates $x_1$ and $x_2$.  $x_1$ is increasing roughly linearly over time, while $x_2$ is changing randomly over time.

# <codecell>

import pymc as mc, pandas as pd, dismod_mr

# <codecell>

np.random.seed(123456)

# <codecell>

# range of time for simulation
years = arange(1990,2011)

# covariates that completely explain variation of prevalence over time
x_1 = mc.rnormal((years-2000) / 10., .2**-2)
x_2 = mc.rnormal(0.*years, .2**-2)

# these covariates change roughly linearly over time
plot(years, x_1, 's-', color=dismod_mr.plot.colors[1], mec='w', ms=10, label='x_1')
plot(years, x_2, 's-', color=dismod_mr.plot.colors[2], mec='w', ms=10, label='x_2')
legend(loc=(1.05,.05));

# <codecell>

# the percent change over time will be a linear combination of x_1 and x_2
plot(years, exp(.5 * x_1 - .25 * x_2), 's-', color=dismod_mr.plot.colors[0], mec='w', ms=10, label='pct change')
legend(loc=(1.05,.05));

# <codecell>

# simulate data
n = 100

data = dict(age=np.random.randint(0, 10, size=n)*10,
            year=np.random.randint(1990, 2010, size=n))
data = pd.DataFrame(data)

data['x_1'] = x_1[array(data.year-1990)]
data['x_2'] = x_2[array(data.year-1990)]

data['value'] = (.1 + .001 * data.age) * exp(.5 * data.x_1 - .25 * data.x_2)
data['data_type'] = 'p'

data['age_start'] = data.age
data['age_end'] = data.age+10

data['area'] = 'all'
data['sex'] = 'total'

data['standard_error'] = -99
data['upper_ci'] = nan
data['lower_ci'] = nan
data['effective_sample_size'] = 1.e8

data.value = clip(data.value, 0, 1)


# build the dismod_mr model
dm = dismod_mr.data.ModelData()

dm.parameters['p'] = dict(parameter_age_mesh=[0,100], level_bounds={'lower': 0.0, 'upper': 1.0},
                          level_value={'age_after': 100, 'age_before': 0, 'value': 0.},
                          heterogeneity='Slightly',
                          fixed_effects={'x_sex': dict(dist='Constant', mu=0)})

# fit and predict full model
dm.input_data = data

# <codecell>

dm.plot()

# <codecell>

dm.setup_model('p', rate_model='neg_binom')

# <codecell>

%time dm.fit(how='mcmc', iter=2000, burn=1000, thin=1)

# <codecell>

dm.plot()
grid();

# <codecell>

dismod_mr.plot.effects(dm, 'p', figsize=(10,2))
xticks(arange(-.25,1,.25))
grid();

# <markdowncell>

# This shows the model is hooked up right... the variation in the data is completely explained by the covariates, and the levels and effect coefficients have been recovered from 100 data points.
# 
# To use the `dm.predict_for` method, we need to fill in values in `dm.output_template`:

# <codecell>

dm.output_template

# <codecell>

dm.output_template = pd.DataFrame(dict(year=years, x_1=x_1, x_2=x_2))
dm.output_template['sex'] = 'total'
dm.output_template['pop'] = 1  # pop is important for aggregating multiple areal units, but not relevant for this case
dm.output_template['area'] = 'all'

# <codecell>

dm.predict_for('all', 'total', 2000).mean()

# <codecell>

dm.plot()
for y in range(1990,2010,2):
    color = cm.spectral((y-1990.)/20.)
    plot(dm.predict_for('all', 'total', y).mean(0), color=color, label=str(y))
    dismod_mr.plot.data_bars(dm.input_data[dm.input_data.year == y], color=color, label='')
legend(loc=(1.05,.05));

# <markdowncell>

# To really see how well this is predicting, we can look at the residuals:

# <codecell>

dismod_mr.plot.plot_one_ppc(dm, 'p')
axis(ymin=-.1, ymax=.1);

# <markdowncell>

# And the output we are really most interested in is how does the predicted percent change year to year compare to the truth

# <codecell>

# the percent change over time will be a linear combination of x_1 and x_2
yy = exp(.5 * x_1 - .25 * x_2)
yy /= yy[0]
plot(years, yy, 's-', color=dismod_mr.plot.colors[0], mec='w', ms=10, label='Truth')
title('Change over time')

yy = array([dm.predict_for('all', 'total', y).mean() for y in years])
yy /= yy[0]
plot(years, yy,
     's-', color=dismod_mr.plot.colors[1], mec='w', ms=10, label='Predicted')

axis(ymin=0)
legend(loc=(1.05,.05));

# <markdowncell>

# What happy results.  Unfortunately, we never have perfect predictors like this.  If we were forced to make due without x_1 or x_2, what would happen?
# 
# For starters, we could have _no_ predictive covariates.  In this case, the predictions for every year will have the same mean, and the uncertainty will be blown up to capture all of the unexplained variation.

# <codecell>

dm.input_data = dm.input_data.drop(['x_1', 'x_2'], axis=1)
dm.setup_model('p', rate_model='neg_binom')
%time dm.fit(how='mcmc', iter=2000, burn=1000, thin=1)

# <codecell>

dm.plot()
for y in range(1990,2011,2):
    plot(dm.predict_for('all', 'total', y).mean(0), color=cm.spectral((y-1990.)/20.), label=str(y))
    
legend(loc=(1.05,.05));

# <markdowncell>

# Now the residuals show the unexplained variation:

# <codecell>

dismod_mr.plot.plot_one_ppc(dm, 'p')
axis(ymin=-.2, ymax=.2);

# <codecell>

# the percent change over time will be a linear combination of x_1 and x_2
yy = exp(.5 * x_1 - .25 * x_2)
yy /= yy[0]
plot(years, yy, 's-', color=dismod_mr.plot.colors[0], mec='w', ms=10, label='Truth')
title('Change over time')

yy = array([dm.predict_for('all', 'total', y).mean() for y in years])
yy /= yy[0]
plot(years, yy,
     's-', color=dismod_mr.plot.colors[1], mec='w', ms=10, label='Predicted')

axis(ymin=0)
legend(loc=(1.05,.05));

# <markdowncell>

# Noting that the values are correlated with time, we could use time as a predictive covariate (appropriately normalized).  This would not be perfect, but it would work pretty well in this case.

# <codecell>

dm.input_data['x_year'] = (data.year-2000.) / 10.
dm.output_template['x_year'] = (years-2000.) / 10.

dm.setup_model('p', rate_model='neg_binom')
%time dm.fit(how='mcmc', iter=2000, burn=1000, thin=1)

# <codecell>

dm.plot()
for y in range(1990,2011,2):
    plot(dm.predict_for('all', 'total', y).mean(0), color=cm.spectral((y-1990.)/20.), label=str(y))
    
legend(loc=(1.05,.05));

# <markdowncell>

# In this case, the residuals are smaller, but not infinitesimal as in the original fit.

# <codecell>

dismod_mr.plot.plot_one_ppc(dm, 'p')
axis(ymin=-.1, ymax=.1);

# <codecell>

# the percent change over time will be a linear combination of x_1 and x_2
yy = exp(.5 * x_1 - .25 * x_2)
yy /= yy[0]
plot(years, yy, 's-', color=dismod_mr.plot.colors[0], mec='w', ms=10, label='Truth')
title('Change over time')

yy = array([dm.predict_for('all', 'total', y).mean() for y in years])
yy /= yy[0]
plot(years, yy,
     's-', color=dismod_mr.plot.colors[1], mec='w', ms=10, label='Predicted')

axis(ymin=0)
legend(loc=(1.05,.05));

# <markdowncell>

# If we had only `x_1`, we could use that as the predictive covariate, and it would work slightly better.

# <codecell>

dm.input_data = dm.input_data.drop(['x_year'], axis=1)
dm.input_data['x_1'] = x_1[array(data.year-1990)]

dm.setup_model('p', rate_model='neg_binom')
%time dm.fit(how='mcmc', iter=2000, burn=1000, thin=1)

# <codecell>

dm.plot()
for y in range(1990,2011,2):
    plot(dm.predict_for('all', 'total', y).mean(0), color=cm.spectral((y-1990.)/20.), label=str(y))
    
legend(loc=(1.05,.05));

# <codecell>

dismod_mr.plot.plot_one_ppc(dm, 'p')
axis(ymin=-.1, ymax=.1);

# <codecell>

# the percent change over time will be a linear combination of x_1 and x_2
yy = exp(.5 * x_1 - .25 * x_2)
yy /= yy[0]
plot(years, yy, 's-', color=dismod_mr.plot.colors[0], mec='w', ms=10, label='Truth')
title('Change over time')

yy = array([dm.predict_for('all', 'total', y).mean() for y in years])
yy /= yy[0]
plot(years, yy,
     's-', color=dismod_mr.plot.colors[1], mec='w', ms=10, label='Predicted')

axis(ymin=0)
legend(loc=(1.05,.05));

# <markdowncell>

# We could also use x_1 and year together, which isn't really to our advantage in this case.

# <codecell>

dm.input_data['x_year'] = (data.year-2000.) / 10.

dm.setup_model('p', rate_model='neg_binom')
%time dm.fit(how='mcmc', iter=2000, burn=1000, thin=1)

# <codecell>

dm.plot()
for y in range(1990,2011,2):
    plot(dm.predict_for('all', 'total', y).mean(0), color=cm.spectral((y-1990.)/20.), label=str(y))
    
legend(loc=(1.05,.05));

# <codecell>

dismod_mr.plot.plot_one_ppc(dm, 'p')
axis(ymin=-.1, ymax=.1);

# <codecell>

# the percent change over time will be a linear combination of x_1 and x_2
yy = exp(.5 * x_1 - .25 * x_2)
yy /= yy[0]
plot(years, yy, 's-', color=dismod_mr.plot.colors[0], mec='w', ms=10, label='Truth')
title('Change over time')

yy = array([dm.predict_for('all', 'total', y).mean() for y in years])
yy /= yy[0]
plot(years, yy,
     's-', color=dismod_mr.plot.colors[1], mec='w', ms=10, label='Predicted')

axis(ymin=0)
legend(loc=(1.05,.05));

# <markdowncell>

# Note that including x_year as a covariate when x_1 is already included provides no additional information.  So anything that looks better in this version is actually *overfitting*.

# <codecell>

!date

# <codecell>


