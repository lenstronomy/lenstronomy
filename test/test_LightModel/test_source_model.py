__author__ = 'sibirrer'

import pytest

from lenstronomy.LightModel.light_model import SourceModel


class TestSourceModel(object):
    """
    tests the source model routines
    """
    def setup(self):
        self.sourceModel = SourceModel(light_model_list=['GAUSSIAN'])
        self.kwargs = [{'amp': 1., 'center_x': 0, 'center_y': 0, 'sigma_x': 2, 'sigma_y': 2}]

    def test_surface_brightness(self):
        output = self.sourceModel.surface_brightness(x=1., y=1., kwargs_source_list=self.kwargs)
        assert output == 0.030987498577413244


if __name__ == '__main__':
    pytest.main()