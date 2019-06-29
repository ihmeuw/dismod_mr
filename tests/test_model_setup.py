"""Test new easy way to setup models."""
import dismod_mr


def test_setup():
    """ Load a model from example dir and set it up with new interface"""
    dm = dismod_mr.load(dismod_mr.testing.get_test_data_dir())

    dm.setup_model(rate_type='p', rate_model='neg_binom')

    # # TODO: test all rate models
    # for rate_model in ['binom', 'log_normal', 'neg_binom',
    #                    'neg_binom_lower_bound', 'neg_binom',
    #                    'normal', 'offest_log_normal', 'poisson']:

    #     dm.setup_model(rate_type='p', rate_model=rate_model)
