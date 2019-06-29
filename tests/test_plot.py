""" Test Plot Module"""

import pandas as pd

tt = pd.DataFrame({'age_start': [0, 10],
                   'age_end': [10, 20],
                   'value': [1, 2],
                   'x_1': [0, 1]
               })

def test_plot_data_bars():
    src.dismod_mr.plot.data_bars(tt)

def test_data_values_by_covariates():
    src.dismod_mr.plot.data_value_by_covariates(tt)
