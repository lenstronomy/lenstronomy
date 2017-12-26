__author__ = 'sibirrer'

import numpy as np
import pytest

from lenstronomy.LightModel.light_model import LensLightModel


class TestLensLightModel(object):
    """
    tests the source model routines
    """
    def setup(self):
        self.lensLightModel = LensLightModel(light_model_list=['GAUSSIAN'])
        self.kwargs = [{'amp': 1., 'center_x': 0., 'center_y': 0., 'sigma_x': 2., 'sigma_y': 2.}]

    def test_surface_brightness(self):
        output = self.lensLightModel.surface_brightness(x=1, y=1, kwargs_lens_light_list=self.kwargs)
        assert output == 0.77880078307140488/(8*np.pi)


if __name__ == '__main__':
    pytest.main()