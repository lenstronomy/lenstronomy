
from lenstronomy.LightModel.linear_basis import LinearBasis


class TestLinearBasis(object):

    def setup_method(self):
        pass

    def test_linear_param_from_kwargs(self):
        linear_basis = LinearBasis(light_model_list=['UNIFORM', 'UNIFORM'])
        kwargs_list = [{'amp': 0.5}, {'amp': -1}]
        param = linear_basis.linear_param_from_kwargs(kwargs_list)
        assert param[0] == kwargs_list[0]['amp']
        assert param[1] == kwargs_list[1]['amp']