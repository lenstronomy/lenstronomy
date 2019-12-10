__author__ = 'sibirrer'

from lenstronomy.LensModel.Profiles.free_form import FreeForm
from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util.util import compute_freeform_alpha

import numpy as np
import numpy.testing as npt
import pytest


class TestFreeForm(object):
    """
    tests the Gaussian methods
    """
    def setup(self):

        self.x_image1, self.y_image1 = 0.8, 0.7
        self.x_image2, self.y_image2 = -0.5, 1.1

        self.x_image, self.y_image = np.array([self.x_image1, self.x_image2]), np.array([self.y_image1, self.y_image2])

        lens_list = ['NFW', 'NFW', 'NFW', 'NFW']
        self.kwargs_list = [{'alpha_Rs': 0.1, 'Rs': 0.1, 'center_x': 0.7, 'center_y':0.2},
                       {'alpha_Rs': 0.2, 'Rs': 0.3, 'center_x': 1.7, 'center_y':-0.2},
                       {'alpha_Rs': 0.07, 'Rs': 0.14, 'center_x': 0.2, 'center_y': -0.9},
                       {'alpha_Rs': 0.08, 'Rs': 0.1, 'center_x': 0.9, 'center_y': 0.4}]
        zlist = [0.5, 0.6, 0.7, 0.8]
        self.source_x, self.source_y = 0.023, -0.04
        self.profile = FreeForm()

        model_list = ['FREEFORM'] + lens_list
        redshift_list = [0.5] + zlist
        self.lensModel = LensModel(model_list, lens_redshift_list=redshift_list, multi_plane=True, z_lens=0.5,
                                   z_source=1.5)

        kwargs = [{'potential': 0, 'alpha_x': 0, 'alpha_y': 0}] + self.kwargs_list
        self.potential_ref = self.lensModel.fermat_potential(self.x_image1, self.y_image1, kwargs)

    def test_solve_alpha(self):

        alpha_x, alpha_y = compute_freeform_alpha(self.source_x, self.source_y, self.x_image1,
                                                  self.y_image1, self.lensModel, self.kwargs_list)

        kwargs = [{'potential': 0, 'alpha_x': alpha_x, 'alpha_y': alpha_y}] + self.kwargs_list

        betax, betay = self.lensModel.ray_shooting(self.x_image1, self.y_image1, kwargs)

        npt.assert_almost_equal(betax, self.source_x, 4)
        npt.assert_almost_equal(betay, self.source_y, 4)

    def test_function(self):

        alpha_x, alpha_y = 0, 0
        kwargs = [{'potential': 1, 'alpha_x': alpha_x, 'alpha_y': alpha_y}] + self.kwargs_list
        potential = self.lensModel.fermat_potential(self.x_image1, self.y_image1, kwargs)
        npt.assert_almost_equal(potential, self.potential_ref - 1)

if __name__ == '__main__':
    pytest.main()
