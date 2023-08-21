from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_size, \
    auto_raytracing_grid_resolution, setup_mag_finite
from lenstronomy.LensModel.lens_model import LensModel
import numpy.testing as npt
import pytest


class TestMagnificationFiniteUtil(object):

    def test_grid_resolution(self):

        source_size = 1.
        res_scale = 0.03
        ref = 15.
        power = 1.2
        resolution = auto_raytracing_grid_resolution(source_size, res_scale, ref, power)

        npt.assert_almost_equal(resolution, res_scale * (source_size/ref) ** power)

    def test_grid_size(self):

        source_size = 20.
        size = auto_raytracing_grid_size(source_size, grid_size_scale=0.04, power=0.9)

        npt.assert_almost_equal(size, 0.04 * source_size ** 0.9)

    def test_setup(self):

        cosmo = None
        lens_model = LensModel(['EPL'])
        grid_radius_arcsec = None
        grid_resolution = None
        source_fwhm_parsec = 30.
        source_light_model = 'SINGLE_GAUSSIAN'
        z_source = 2.
        source_x, source_y = 0., 0.
        dx, dy, amp_scale, size_scale = None, None, None, None
        gridx, gridy, source_model, kwargs_source, grid_resolution, grid_radius_arcsec = setup_mag_finite(cosmo, lens_model, grid_radius_arcsec, grid_resolution,
                                                                                                          source_fwhm_parsec, source_light_model,
                                                                                                          z_source, source_x, source_y, dx, dy,
                                                                                                          amp_scale, size_scale)
        npt.assert_equal(True, len(source_model.func_list)==1)
        npt.assert_equal(True, grid_resolution is not None)
        npt.assert_equal(True, grid_radius_arcsec is not None)

        grid_resolution = 0.001
        grid_radius_arcsec = 0.05
        dx, dy, amp_scale, size_scale = 0., 0.1, 1., 1.
        source_light_model = 'DOUBLE_GAUSSIAN'
        gridx, gridy, source_model, kwargs_source, grid_resolution, grid_radius_arcsec = setup_mag_finite(cosmo, lens_model, grid_radius_arcsec, grid_resolution,
                                                                                                          source_fwhm_parsec, source_light_model,
                                                                                                          z_source, source_x, source_y, dx, dy,
                                                                                                          amp_scale, size_scale)
        npt.assert_equal(True, len(source_model.func_list) == 2)
        npt.assert_equal(kwargs_source[1]['center_y'], kwargs_source[0]['center_y'] + dy)
        npt.assert_equal(kwargs_source[1]['center_x'], kwargs_source[0]['center_x'] + dx)
        npt.assert_equal(True, grid_resolution == 0.001)
        npt.assert_equal(True, grid_radius_arcsec == 0.05)

        source_light_model = 'trash'
        npt.assert_raises(Exception, setup_mag_finite,
                          cosmo,
                          lens_model,
                          grid_radius_arcsec,
                          grid_resolution,
                          source_fwhm_parsec,
                          source_light_model,
                          z_source,
                          source_x,
                          source_y, dx,
                          dy,
                          amp_scale,
                          size_scale
                          )


if __name__ == '__main__':
    pytest.main()
