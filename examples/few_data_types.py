# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Consistent models in DisMod-MR without many different types of data
# 
# In DisMod-II there was a requirement to have at least three different data types, corresponding to different parts of the compartmental model.  DisMod-MR can run a compartmental model with only two, or even one data type, but this requires expert priors to fill in the gaps.
# 
# This document provides an example of how a model for Parkinson's Disease might look with different subsets of data types.

# <codecell>

import dismod_mr

# <codecell>

models = {}
#iter=101; burn=0; thin=1  # use these settings to run faster
iter=10000; burn=5000; thin=5  # use these settings to make sure MCMC converges

# <markdowncell>

# # Consistent fit with all data
# 
# Let's start with a consistent fit of the simulated PD data.  This includes data on prevalence, incidence, and SMR, and the assumption that remission rate is zero.  All together this counts as four different data types in the DisMod-II accounting.

# <codecell>

model = dismod_mr.load('/homes/abie/notebook/pd_sim_data/')
model.keep(areas=['GBR'], sexes=['female', 'total'])

# <codecell>

model.setup_model()
%time model.fit(iter=iter, burn=burn, thin=thin)

# <codecell>

models['p, i, r, smr'] = model
model.plot()

# <markdowncell>

# # Consistent fit without incidence
# 
# Now let's do it again with the incidence removed.  Since there is data on prevalence and SMR as well as the assumption that remission is zero, this counts as three data types, the minimum allowed for DisMod-II.

# <codecell>

model = dismod_mr.load('/homes/abie/notebook/pd_sim_data/')
model.keep(areas=['GBR'], sexes=['female', 'total'])

model.input_data = model.input_data[model.input_data.data_type != 'i']
print 'kept %d rows' % len(model.input_data.index)

# <codecell>

model.setup_model()
%time model.fit(iter=iter, burn=burn, thin=thin)

# <codecell>

models['p, r, smr'] = model
model.plot()

# <markdowncell>

# # Consistent fit without incidence or mortality
# 
# This uses only prevalence data and the assumption that there is no remission, so it is not valid in DisMod-II.  The Bayesian priors included by default in DisMod-MR make it possible, but the tradeoff between incidence and mortality is not informed by any data in this case.

# <codecell>

model = dismod_mr.load('/homes/abie/notebook/pd_sim_data/')
model.keep(areas=['GBR'], sexes=['female', 'total'])

model.input_data = model.input_data[model.input_data.data_type == 'p']
print 'kept %d rows' % len(model.input_data.index)

# <codecell>

model.setup_model()
%time model.fit(iter=iter, burn=burn, thin=thin)

# <codecell>

models['p, r'] = model
model.plot()

# <markdowncell>

# # Consistent fit with only prevalence
# 
# Now without assumption of zero remission, DisMod-MR is going for a very underconstrained problem, and relies on the priors heavily.  However, the prevalence data is there, so the estimates of prevalence will not be changed much.

# <codecell>

model = dismod_mr.load('/homes/abie/notebook/pd_sim_data/')
model.keep(areas=['GBR'], sexes=['female', 'total'])

model.input_data = model.input_data[model.input_data.data_type == 'p']
print 'kept %d rows' % len(model.input_data.index)

model.set_level_bounds('r', 0., 1.)
model.set_level_value('r', age_before=0., age_after=101., value=0)

# <codecell>

model.setup_model()
%time model.fit(iter=iter, burn=burn, thin=thin)

# <codecell>

models['p'] = model
model.plot()

# <markdowncell>

# # Comparison of alternative models
# 
# Let's compare the distributions for all of these now.  You can see that the more data there is, the more concentrated the posterior distribution becomes.

# <codecell>

for i, (label, model) in enumerate(models.items()):
    hist(model.vars['p']['mu_age'].trace().mean(1), normed=True, histtype='step',
         color=dismod_mr.plot.colors[i%4], linewidth=3, linestyle=['solid','dashed'][i/4],
         label=label)
legend(loc=(1.1,.1))
title('Posterior Distribution Comparison\nCrude Prevalence');

# <codecell>

for i, (label, model) in enumerate(models.items()):
    hist(model.vars['i']['mu_age'].trace().mean(1), normed=True, histtype='step',
         color=dismod_mr.plot.colors[i%4], linewidth=3, linestyle=['solid','dashed'][i/4],
         label=label)
legend(loc=(1.1,.1)),
title('Posterior Distribution Comparison\nCrude Incidence');

# <markdowncell>

# # Consistent fit without prevalence
# 
# The really challenging case is without any prevalence data. DisMod-MR will go for it, but there will be a lot of uncertainty.

# <codecell>

model = dismod_mr.load('/homes/abie/notebook/pd_sim_data/')
model.keep(areas=['GBR'], sexes=['female', 'total'])

model.input_data = model.input_data[model.input_data.data_type != 'p']
print 'kept %d rows' % len(model.input_data.index)

model.setup_model()
%time model.fit(iter=iter, burn=burn, thin=thin)

# <codecell>

models['i, r, smr'] = model
model.plot()

# <markdowncell>

# # Consistent fit with incidence only
# 
# DisMod-MR it will even go for it with _only_ incidence.  But that is not ideal...

# <codecell>

model = dismod_mr.load('/homes/abie/notebook/pd_sim_data/')
model.keep(areas=['GBR'], sexes=['female', 'total'])

model.input_data = model.input_data[model.input_data.data_type == 'i']
print 'kept %d rows' % len(model.input_data.index)
model.set_level_bounds('r', 0., 1.)

model.setup_model()
%time model.fit(iter=iter, burn=burn, thin=thin)

# <codecell>

models['i'] = model
model.plot()

# <markdowncell>

# # Consistent fit without prevalence or incidence
# 
# DisMod-MR is not magic, however.  Without prevalence _or_ incidence, it will not know how much PD there is!

# <codecell>

model = dismod_mr.load('/homes/abie/notebook/pd_sim_data/')
model.keep(areas=['GBR'], sexes=['female', 'total'])

model.input_data = model.input_data[model.input_data.data_type == 'smr']
print 'kept %d rows' % len(model.input_data.index)

model.setup_model()
%time model.fit(iter=iter, burn=burn, thin=thin)

# <codecell>

models['r, smr'] = model
model.plot()

# <codecell>

for i, label in enumerate(['p', 'i, r, smr', 'i', 'r, smr']):
    hist(models[label].vars['p']['mu_age'].trace().mean(1), normed=False, histtype='step',
         color=dismod_mr.plot.colors[i%4], linewidth=3, linestyle=['solid','dashed'][i/4],
         label=label)
legend(loc=(1.1,.1))
title('Posterior Distribution Comparison\nCrude Prevalence')
axis(xmin=-.001);

# <codecell>

!date

# <codecell>


