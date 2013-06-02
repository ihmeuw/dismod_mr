""" Test Data Model """

import pylab as pl
import pymc as mc

from dismod_mr import data
reload(data)

def test_blank_input_data():
    d = data.ModelData()

    for field in 'data_type value area sex age_start age_end year_start year_end standard_error effective_sample_size lower_ci upper_ci age_weights'.split():
        assert field in d.input_data.columns, 'Input data CSV should have field "%s"' % field

    for field in 'data_type area sex year pop'.split():
        assert field in d.output_template.columns, 'Output template CSV should have field "%s"' % field

    for data_type in 'i p r f rr X ages'.split():
        assert data_type in d.parameters, 'Parameter dict should have entry for "%s"' % data_type

    assert d.hierarchy.number_of_nodes() > 0, 'Hierarchy should be non-empty'

    assert len(d.nodes_to_fit) > 0, 'Nodes to fit should be non-empty'


if __name__ == '__main__':
    import nose
    nose.runmodule()
    
