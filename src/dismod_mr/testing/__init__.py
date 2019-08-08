import os

from dismod_mr.testing import data_simulation


def get_test_data_dir():
    return os.path.dirname(os.path.abspath(__file__)) + '/example_data'
