# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Getting estimates out of DisMod-MR
# 
# The goal of this document is to demonstrate how to export age-specific prevalence estimates from DisMod-MR in a comma-separated value (CSV) format, for use in subsequent analysis.
# 
# It uses data from the replication dataset for regional estimates of HCV prevalence, as published in Mohd Hanafiah K, Groeger J, Flaxman AD, Wiersma ST. Global epidemiology of hepatitis C virus infection: New estimates of age-specific antibody to HCV seroprevalence. Hepatology. 2013 Apr;57(4):1333-42. doi: 10.1002/hep.26141. Epub 2013 Feb 4.  http://www.ncbi.nlm.nih.gov/pubmed/23172780
# 
# The dataset is available from: http://ghdx.healthmetricsandevaluation.org/record/hepatitis-c-prevalence-1990-and-2005-all-gbd-regions
# 
#     wget http://ghdx.healthmetricsandevaluation.org/sites/ghdx/files/record-attached-files/IHME_GBD_HEP_C_RESEARCH_ARCHIVE_Y2013M04D12.ZIP
#     unzip IHME_GBD_HEP_C_RESEARCH_ARCHIVE_Y2013M04D12.ZIP

# <codecell>

#!wget http://ghdx.healthmetricsandevaluation.org/sites/ghdx/files/record-attached-files/IHME_GBD_HEP_C_RESEARCH_ARCHIVE_Y2013M04D12.ZIP
#!unzip IHME_GBD_HEP_C_RESEARCH_ARCHIVE_Y2013M04D12.ZIP

# <codecell>

# This Python code will export predictions 
# for the following region/sex/year:
predict_region = 'asia_central'
predict_sex = 'male'
predict_year = 2005

# <codecell>

# import dismod code
import dismod_mr

# <markdowncell>

# Load the model, and keep only data for the prediction region/sex/year

# <codecell>

model_path = 'hcv_replication/'
dm = dismod_mr.data.load(model_path)

if predict_year == 2005:
    dm.keep(areas=[predict_region], sexes=['total', predict_sex], start_year=1997)
elif predict_year == 1990:
    dm.keep(areas=[predict_region], sexes=['total', predict_sex], end_year=1997)
else:
    raise Error, 'predict_year must equal 1990 or 2005'

# <codecell>

# Fit model using the data subset (faster, but no borrowing strength)
dm.vars += dismod_mr.model.process.age_specific_rate(dm, 'p', predict_region, predict_sex, predict_year)
%time dismod_mr.fit.asr(dm, 'p', iter=2000, burn=1000, thin=1)

# <codecell>

# Make posterior predictions
pred = dismod_mr.model.covariates.predict_for(
            dm, dm.parameters['p'],
            predict_region, predict_sex, predict_year,
            predict_region, predict_sex, predict_year, True, dm.vars['p'], 0, 1)

# <markdowncell>

# The easiest way to get these predictions into a csv file is to use the Python Pandas package:

# <codecell>

import pandas as pd

# <codecell>

# This generates a csv with 1000 rows,
# one for each draw from the posterior distribution

# Each column corresponds to a one-year age group,
# e.g. column 10 is prevalence at age 10

pd.DataFrame(pred).to_csv(
    model_path + '%s-%s-%s.csv'%(predict_region, predict_sex, predict_year))

# <codecell>

!ls -hal hcv_replication/asia_central-male-2005.csv

# <markdowncell>

# To aggregate this into pre-specified age categories, you need to specify the age weights and groups:

# <codecell>

weights = [1, 8, 8, 9, 9, 10, 10, 10, 10, 10,
           10, 10, 10, 10, 9, 9, 9, 9, 9, 9,
           9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
           9, 9, 8, 8, 8, 8, 8, 8, 8, 8,
           8, 7, 7, 7, 7, 7, 7, 7, 7, 7,
           6, 6, 6, 6, 6, 6, 5, 5, 5, 5,
           5, 5, 4, 4, 4, 4, 4, 4, 4, 3,
           3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
           3, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# <codecell>

# 1000 samples from the posterior distribution for age-standardized prevalence

age_std = np.dot(pred, weights) / np.sum(weights)
hist(age_std, color='#cccccc', normed=True)
xlabel('Age-standardized Prevalence')
ylabel('Posterior Probability');

# <markdowncell>

# You can extract an age-standardized point and interval estimate from the 1000 draws from the posterior distribution stored in age_std as follows:  (to do this for crude prevalence, use the population weights to average, instead of standard weights.)

# <codecell>

import pymc as mc

print 'age_std prev mean:', age_std.mean()
print 'age_std prev 95% UI:', mc.utils.hpd(age_std, .05)

# <markdowncell>

# For groups, just do the same thing group by group:

# <codecell>

group_cutpoints = [0, 1,  5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 100]

# <codecell>

results = []
for a0, a1 in zip(group_cutpoints[:-1], group_cutpoints[1:]):
    age_grp = np.dot(pred[:, a0:a1], weights[a0:a1]) / np.sum(weights[a0:a1])
    results.append(dict(a0=a0,a1=a1,mu=age_grp.mean(),std=age_grp.std()))
    
results = pd.DataFrame(results)
print np.round(results.head(), 2)

# <codecell>

errorbar(.5*(results.a0+results.a1), results.mu,
         xerr=.5*(results.a1-results.a0),
         yerr=1.96*results['std'],
         fmt='ks', capsize=0, mec='w')
axis(ymin=0, xmax=100)

# <codecell>

!date

# <codecell>


