# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# A motivating example: descriptive epidemiological meta-regression of Parkinson's Disease
# ========================================================================================
# 
# The goal of this document it give a concise demonstration of the 
# strengths and limitations of DisMod-MR, the descriptive
# epidemiological meta-regression tool developed for the Global Burden of Disease,
# Injuries, and Risk Factors 2010 (GBD2010) Study.

# <markdowncell>

# A systematic review of PD was conducted as part of the GBD 2010
# Study. The results of this
# review---data on the prevalence, incidence, and standardized mortality ratio of
# PD---needed to be combined to produce estimates of disease prevalence by
# region, age, sex, and year.  These prevalence estimates were combined
# with disability weights to measure years lived with disability (YLDs),
# which were then combined with estimates of years of life lost (YLLs)
# to produce estimates of the burden of PD quantified in disability-adjusted life-years (DALYs).
# 
# PD is a neurodegenerative disorder that includes symptoms of motor
# dysfunction, such as tremors, rigidity, and akinesia, in the early
# stages of the disease.  As the disease develops, most patients also
# develop nonmotor symptoms, such as cognitive decline, dementia,
# autonomic failure, and disordered sleep-wake regulation.  The standard
# definition for PD diagnosis includes at least two of four cardinal
# signs---resting tremor, bradykinesia, rigidity, and postural abnormalities.
# There is no cure or treatments to slow the progression of the disease;
# however, motor symptoms and disability may be improved with
# symptomatic therapy.

# <markdowncell>

# This document works with simulated data, so that the dataset is fully distributable.  This data is included for understanding how to use DisMod-MR, and is not intended for the study of the descriptive epidemiology of PD.

# <codecell>

import dismod_mr
import pymc as mc, pandas as pd

# <markdowncell>

# 
# DisMod-MR uses the integrative systems modeling (ISM) approach to produce simultaneous
# estimates of disease incidence, prevalence, remission, and mortality. The hallmark of
# ISM is incorporating all available data.  In the case of Parkinson's Disease this
# consists of population level measurements of incidence, prevalence, standardized mortality rate (SMR),
# and cause-specific mortality rate (CSMR).
# 
# I will begin with a look at a subset of this data, however.  Only that from females in the Europe, Western GBD region.

# <codecell>

model = dismod_mr.data.load('pd_sim_data')
model.keep(areas=['europe_western'], sexes=['female', 'total'])

# <markdowncell>

# Of the 348 rows of data, here is how the values breakdown by data type:

# <codecell>

summary = model.input_data.groupby('data_type')['value'].describe()
round_(summary,3).unstack().sort('count', ascending=False)

# <markdowncell>

# More than half of the available data for this region is prevalence data.  I'll take a closer look
# at that now.

# <codecell>

model.get_data('smr').value.mean()

# <codecell>

groups = model.get_data('p').groupby('area')
print round_(groups['value'].describe(),3).unstack().sort('50%', ascending=False)

# <markdowncell>

# In the original dataset, there was a wide range in median values, which reflects a combination of country-to-country variation and compositional bias.  Simulating data has reduced this substantially, but there is still six-fold variation between ESP and GBR.

# <codecell>

countries = ['ESP', 'GBR']
c = {}
for i, c_i in enumerate(countries):
    c[i] = groups.get_group(c_i)

# <codecell>

ax = None
figure(figsize=(10,4))
for i, c_i in enumerate(countries):
    ax = subplot(1,2,1+i, sharey=ax, sharex=ax)
    dismod_mr.plot.data_bars(c[i])
    xlabel('Age (years)')
    ylabel('Prevalence (per 1)')
    title(c_i)
axis(ymin=-.001, xmin=-5, xmax=105)
subplots_adjust(wspace=.3)

# <markdowncell>

# A model for age-specific parameters when measurements have heterogeneous age groups
# -----------------------------------------------------------------------------------

# <markdowncell>

# DisMod-MR has four features that make it particularly suited for estimating age-specific prevalence of PD from this data:
# 
# * Piecewise linear spline model for change in prevalence as a function of age
# * Age-standardizing model of age-group heterogeneity represents the heterogeneous age groups collected in systematic review
# * Country-level random effects for true variation in prevalence between countries
# * Negative binomial model of data, which provides data-driven estimation of non-sampling error in measurements
#   and elegantly handles measurements of 0

# <markdowncell>

# I will now fit the prevalence data with DisMod-MR's age-standardizing negative binomial random effect spline model and
# compare the estimates to the observed data.  Then I will use the results of the fit model to explore the four features listed above.

# <codecell>

# remove fixed effects for this example, I will return to them below
model.input_data = model.input_data.filter(regex='(?!x_)')

# <codecell>

model.vars += dismod_mr.model.asr(model, 'p')
%time dismod_mr.fit.asr(model, 'p')

# <codecell>

# plot age-specific prevalence estimates over data bars
figure(figsize=(10,4))

dismod_mr.plot.data_bars(model.get_data('p'), color='grey', label='Simulated PD Data')
pred = dismod_mr.model.predict_for(model, model.parameters['p'], 'all', 'female', 2005,
                                      'europe_western', 'female', 2005, 1.,
                                      model.vars['p'], 0., 1.)    # TODO: simplify this method!

hpd = mc.utils.hpd(pred, .05)

plot(arange(101), pred.mean(axis=0), 'k-', linewidth=2, label='Posterior Mean')
plot(arange(101), hpd[:,0], 'k--', linewidth=1, label='95% HPD interval')
plot(arange(101), hpd[:,1], 'k--', linewidth=1)

xlabel('Age (years)')
ylabel('Prevalence (per 1)')
grid()
legend(loc='upper left')

axis(ymin=-.001, xmin=-5, xmax=105)

# <codecell>

p_only = model  # store results for future comparison

# <markdowncell>

# This estimate shows the nonlinear increase in prevalence as a function of age, where the slope of the
# curve increases at age 60.  A nonlinear estimate like this is possible thanks to DisMod-MR's piecewise linear
# spline model.
# 
# The age-standardizing model for heterogeneous age groups is also important for
# such settings; a naive approach, such as using the age interval midpoint, would result in under-estimating
# the prevalence for age groups that include both individuals older and younger than 60.

# <markdowncell>

# 
# The exact age where the slope of the curve changes is _not_ entirely data driven in this example.  The knots
# in the piecewise linear spline model were chosen a priori, on the following grid:

# <codecell>

model.parameters['p']['parameter_age_mesh']

# <markdowncell>

# A sparse grid allows faster computation, but a dense grid allows more expressive age pattens.  Choosing
# the proper balance is one challenge of a DisMod-MR analysis.  This is especially true for sparse,
# noisy data, where too many knots allow the model to follow noisy idiosyncrasies of the data.  DisMod-MR
# allows for penalized spline regression to help with this choice.

# <markdowncell>

# The country-level random effects in this model capture country-to-country variation in PD prevalence.
# This variation is not visible in the graphic above, which shows the regional aggregation of country-level
# estimates (using a population weighted average that takes uncertainty into account).
# 
# The country-level random effects take the form of intercept shifts in log-prevalence space, with values
# showing in the following:

# <codecell>

df = pd.DataFrame(index=[alpha_i.__name__ for alpha_i in model.vars['p']['alpha']],
                      columns=['mean', 'lb', 'ub'])
for alpha_i in model.vars['p']['alpha']:
    stats = alpha_i.stats()
    df.ix[alpha_i.__name__] = (stats['mean'], stats['95% HPD interval'][0], stats['95% HPD interval'][1])

# <codecell>

print round_(df.astype(float),3).sort('mean', ascending=False)

# <markdowncell>

# The fourth feature of the model which I want to draw attention to here is the negative binomial model of data,
# which deals with measurements of zero prevalence in a principled way.  Prevalence studies are reporting transformations
# of count data, and count data can be zero.  In the case of prevalence of PD in 30- to 40-year-olds, it often _will_ be zero.

# <codecell>

model.get_data('p').sort('age_start').filter(['age_start', 'age_end', 'area', 'value']).head(15)

# <markdowncell>

# The negative binomial model has an appropriately skewed distribution, where prevalence measurements 
# of zero are possible, but measurements of less than zero are not possible.  To demonstrate how this
# functions, the next figure shows the "posterior predictive distribution" for the measurements above,
# i.e. sample values that the model predicts would be found of the studies were conducted again under
# the same conditions.

# <codecell>

pred = model.vars['p']['p_pred'].trace()
obs = array(model.vars['p']['p_obs'].value)
ess = array(model.vars['p']['p_obs'].parents['n'])

# <codecell>

figure(figsize=(10,4))

sorted_indices = obs.argsort().argsort()
jitter = mc.rnormal(0, .1**-2, len(pred))

for i,s_i in enumerate(sorted_indices):
    plot(s_i+jitter, pred[:, i], 'ko', mew=0, alpha=.25, zorder=-99)

errorbar(sorted_indices, obs, yerr=1.96*sqrt(obs*(1-obs)/ess), fmt='ks', mew=1, mec='white', ms=5)

xticks([])
xlabel('Measurement')
ylabel('Prevalence (%)\n', ha='center')
yticks([0, .02, .04, .06, .08], [0, 2, 4, 6, 8])
axis([25.5,55.5,-.01,.1])
grid()
title('Posterior Predictive distribution')

# <markdowncell>

# Additional features of DisMod-MR
# --------------------------------

# <markdowncell>

# Four additional features of DisMod-MR that are important for many settings are:
# 
# * informative priors
# * fixed effects to cross-walk between different studies
# * fixed effects to predict out of sample
# * fixed effects to explain the level of variation

# <markdowncell>

# Informative priors are useful for modeling disease with less data available than PD, for example to include
# information that prevalence is zero for youngest ages, or than prevalence must be increasing as a function of
# age between certain ages.
# 
# The informative priors are also key to the "empirical Bayes" approach to modeling age-specific differences between
# difference GBD regions.  In this setting, a model using all the world's data is used to produce estimates for each region,
# and these estimates are used as priors in region-specific models together with the data relevant to that region only.

# <markdowncell>

# "Cross-walk" fixed effects can correct for biases introduced by multiple outcome measures.  For example, in the PD dataset,

# <codecell>

model = dismod_mr.data.load('pd_sim_data')

# <codecell>

crosswalks = list(model.input_data.filter(like='x_cv').columns)
groups = model.get_data('p').groupby(crosswalks)

# <codecell>

crosswalks

# <codecell>

round_(groups['value'].describe(),3).unstack()['mean']

# <markdowncell>

# Incorporating data on parameters other than prevalence
# ------------------------------------------------------
# 
# So far this example has focused on modeling the prevalence of PD from the
# prevalence data alone.  However, this represents about half of the available
# data.  There is also information on incidence, SMR, and CSMR, which has not
# yet been incorporated.
# 
# DisMod-MR is capable of including all of the available data, using a compartmental
# model of disease moving through a population.  This model formalizes the observation
# that prevalent cases must once have been incident cases, and continue to be prevalent
# cases until remission or death.
# 
# In this model, incidence, remission, and excess-mortality are age-standardizing negative binomial random effect spline models,
# while prevalence, SMR, CSMR, and other parameters come from the solution to a system of ordinary differential equations.
# 
# The results of this model are smoother prevalence curves that take longer to calculate.

# <codecell>

figure(figsize=(10,6))

subplot(2,2,1); dismod_mr.plot.data_bars(model.get_data('p')); xlabel('Age (years)'); ylabel('Prevalence')
subplot(2,2,2); dismod_mr.plot.data_bars(model.get_data('i')); xlabel('Age (years)'); ylabel('Incidence')
subplot(2,2,3); dismod_mr.plot.data_bars(model.get_data('csmr')); xlabel('Age (years)'); ylabel('Cause-specific mortality')
subplot(2,2,4); dismod_mr.plot.data_bars(model.get_data('smr')); xlabel('Age (years)'); ylabel('Standardized \nmortality ratio')

# <codecell>

model.input_data.columns

# <codecell>

model.vars += dismod_mr.model.consistent(model)
%time dismod_mr.fit.consistent(model)

# <codecell>

figure(figsize=(10,6))

subplot(2,2,1); dismod_mr.plot.data_bars(model.get_data('p')); xlabel('Age (years)'); ylabel('Prevalence')
subplot(2,2,2); dismod_mr.plot.data_bars(model.get_data('i')); xlabel('Age (years)'); ylabel('Incidence')
subplot(2,2,3); dismod_mr.plot.data_bars(model.get_data('csmr')); xlabel('Age (years)'); ylabel('Cause-specific mortality')
subplot(2,2,4); dismod_mr.plot.data_bars(model.get_data('smr')); xlabel('Age (years)'); ylabel('Standardized \nmortality ratio')
param_list = [dict(type='p', title='(a)', ylabel='Prevalence (%)', yticks=([0, .01, .02], [0, 1, 2]), axis=[30,101,-0.001,.025]),
          dict(type='i', title='(b)', ylabel='Incidence \n(per 1000 PY)', yticks=([0, .001,.002, .003, .004], [0, 1, 2, 3, 4]), axis=[30,104,-.0003,.0055]),
          dict(type='pf', title='(c)', ylabel='Cause-specific mortality \n(per 1000 PY)', yticks=([0, .001,.002], [0, 1, 2]), axis=[30,104,-.0002,.003]),
          dict(type='smr', title='(d)', ylabel='Standardized \nmortality ratio', yticks=([1, 2, 3,4, ], [1, 2,3, 4]), axis=[35,104,.3,4.5]),
          ]

for i, params in enumerate(param_list):
    ax = subplot(2,2,i+1)
    if params['type'] == 'pf': dismod_mr.plot.data_bars(model.get_data('csmr'), color='grey')
    else: dismod_mr.plot.data_bars(model.get_data(params['type']), color='grey')
    
    if params['type'] == 'smr': model.pred = dismod_mr.model.predict_for(model, model.parameters.get('smr', {}), 'all', 'female', 2005, 
                                                               'europe_western', 'female', 2005, 1., model.vars['smr'], 0., 100.).T
    else : model.pred = dismod_mr.model.predict_for(model, model.parameters[params['type']],
                                                       'all', 'female', 2005, 
                                                       'europe_western', 'female', 2005, 1., model.vars[params['type']], 0., 1.).T
    
    plot(arange(101), model.pred.mean(axis=1), 'k-', linewidth=2, label='Posterior Mean')
    hpd = mc.utils.hpd(model.pred.T, .05)
    plot(arange(101), hpd[:,0], 'k-', linewidth=1, label='95% HPD interval')
    plot(arange(101), hpd[:,1], 'k-', linewidth=1)

    xlabel('Age (years)')
    ylabel(params['ylabel']+'\n\n', ha='center')
    axis(params.get('axis', [-5,105,-.005,.06]))
    yticks(*params.get('yticks', ([0, .025, .05], [0, 2.5, 5])))
    title(params['title'])
    grid()
    
subplots_adjust(hspace=.35, wspace=.35, top=.97)

# <codecell>

p_with = model

# <markdowncell>

# The most notable difference between the estimates from this model and from the model
# that used prevalence data only is that this model produces estimates of incidence and
# mortality in addition to prevalence.  In many cases, the model also produces estimates
# of the remission rate as well, but there is no remission of PD, so the estimates of zero
# are not very interesting in this example.  It is another place that informative priors are useful,
# however.
# 
# There are also differences between the means and uncertainty intervals estimated by these methods,
# which show that the additional data is important.  Although the prevalence data alone predicts 
# age-specific prevalence that peaks at 2%, when the incidence and mortality data is also included, the
# maximum prevalence is a bit lower, closer to 1.5%.

# <codecell>

p1 = dismod_mr.model.predict_for(p_only, model.parameters['p'],
                                    'all', 'total', 'all', 
                                    'europe_western', 'female', 2005, 1.,
                                    p_only.vars['p'], 0., 1.)

p2 = dismod_mr.model.predict_for(p_with, model.parameters['p'],
                                    'all', 'total', 'all', 
                                    'europe_western', 'female', 2005, 1.,
                                    p_with.vars['p'], 0., 1.)

# <codecell>

plot(p1.mean(axis=0), 'k--', linewidth=2, label='Only prevalence')
plot(p2.mean(axis=0), 'k-', linewidth=2, label='All available')

xlabel('Age (years)')
ylabel('Prevalence (%)\n\n', ha='center')
yticks([0, .01, .02], [0, 1, 2])
axis([30,101,-0.001,.025])
legend(loc='upper left')
grid()

subplots_adjust(top=.97, bottom=.16)

# <markdowncell>

# Because the data is so noisy, the differences between the mean estimates of these different models are not significant; the posterior distributions
# have considerable overlap.  At age 80, for example, the posterior distributions for age-80 prevalence are estimated as the following:

# <codecell>

hist(100*p1[:,80], normed=True, histtype='step', label='Only prevalence', linewidth=3, color=array([239., 138., 98., 256.])/256)
hist(100*p2[:,80], normed=True, histtype='step', label='All available', linewidth=3, color=array([103, 169, 207, 256.])/256)
title('PD prevalence at age 80')
xlabel('Prevalence (%)\n\n', ha='center')
ylabel('Probability Density')
legend(loc='upper right')
grid()

subplots_adjust(bottom=.16)

# <markdowncell>

# Conclusion
# ==========
# 
# I hope that this example is a quick way to see the strengths and weaknesses of DisMod-MR.
# This model is particularly suited for estimating descriptive epidemiology of diseases
# with sparse, noisy data from multiple, incompatible sources.
# 
# I am currently working to make it faster, as well as to improve the capabilities for modeling
# changes between regions over time.

# <codecell>

!date

# <codecell>


