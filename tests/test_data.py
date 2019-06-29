"""Test Data Model."""
import dismod_mr


def test_blank_input_data():
    d = dismod_mr.data.ModelData()

    for field in ('data_type value area sex age_start age_end year_start year_end '
                  'standard_error effective_sample_size lower_ci upper_ci age_weights').split():
        assert field in d.input_data.columns, f'Input data CSV should have field "{field}"'

    for field in 'data_type area sex year pop'.split():
        assert field in d.output_template.columns, f'Output template CSV should have field "{field}"'

    for data_type in 'i p r f rr X ages'.split():
        assert data_type in d.parameters, f'Parameter dict should have entry for "{data_type}"'

    assert d.hierarchy.number_of_nodes() > 0, 'Hierarchy should be non-empty'

    assert len(d.nodes_to_fit) > 0, 'Nodes to fit should be non-empty'


def test_set_effect_prior():
    dm = dismod_mr.data.ModelData()
    dm.set_effect_prior('p', 'x_sex', dict(dist='Constant', mu=.1))
