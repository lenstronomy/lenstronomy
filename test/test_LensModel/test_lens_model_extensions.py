__author__ = 'sibirrer'

import numpy.testing as npt
import pytest
import astrofunc.util as util
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.lens_model import LensModel


class TestLensModel(object):
    """
    tests the source model routines
    """
    def setup(self):
        pass

    def test_critical_curves(self):
        lens_model_list = ['SPEP']
        deltaPix = 0.05
        numPix = 100
        x_grid, y_grid, x_0, y_0, ra_0, dec_0, Matrix, Matrix_inv = util.make_grid_with_coordtransform(numPix=numPix,
                                                                                                       deltapix=deltaPix,
                                                                                                       subgrid_res=1)
        kwargs_lens = [{'theta_E': 1., 'gamma': 2., 'q': 0.8, 'phi_G': 1., 'center_x': 0, 'center_y': 0}]
        lensModel = LensModelExtensions(lens_model_list)
        ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list = lensModel.critical_curve_caustics(kwargs_lens,
                                                                                                 kwargs_else={}, compute_window=5, grid_scale=0.005)
        import matplotlib.pyplot as plt
        lensModel = LensModel(lens_model_list)
        x_grid_high_res, y_grid_high_res = util.make_subgrid(x_grid, y_grid, 10)
        mag_high_res = util.array2image(
            lensModel.magnification(x_grid_high_res, y_grid_high_res, kwargs_lens, kwargs_else={}))

        cs = plt.contour(util.array2image(x_grid_high_res), util.array2image(y_grid_high_res), mag_high_res, [0],
                         alpha=0.0)
        paths = cs.collections[0].get_paths()
        for i, p in enumerate(paths):
            v = p.vertices
            ra_points = v[:, 0]
            dec_points = v[:, 1]
            print(ra_points, ra_crit_list[i])
            npt.assert_almost_equal(ra_points[0], ra_crit_list[i][0], 5)
            npt.assert_almost_equal(dec_points[0], dec_crit_list[i][0], 5)


if __name__ == '__main__':
    pytest.main("-k TestLensModel")



