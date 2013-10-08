""" Test new easy way to setup models
"""

import dismod_mr


def test_setup():
    """ Load a model from example dir and set it up with new interface"""
    import os
    test_dir = os.path.dirname(os.path.abspath(__file__))

    dm = dismod_mr.load(test_dir + '/example_data')

    dm.setup_model(rate_type='p', rate_model='neg_binom')

    # # TODO: test all rate models
    # for rate_model in ['binom', 'log_normal', 'neg_binom',
    #                    'neg_binom_lower_bound', 'neg_binom',
    #                    'normal', 'offest_log_normal', 'poisson']:

    #     dm.setup_model(rate_type='p', rate_model=rate_model)

if __name__ == '__main__':
    import nose
    nose.runmodule()
    
