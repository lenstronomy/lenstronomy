import pytest
import numpy as np
import numpy.testing as npt
import copy

from lenstronomy.Data.imaging_data import Data
import lenstronomy.Util.util as util


class TestData(object):
    def setup(self):
        self.numPix = 10
        kwargs_data = {'image_data': np.zeros((self.numPix, self.numPix))}
        self.Data = Data(kwargs_data)

    def test_get_covariance_matrix(self):
        d = np.array([1, 2, 3])
        sigma_b = 1
        f = 10.
        result = self.Data.covariance_matrix(d, sigma_b, f)
        assert result[0] == 1.1
        assert result[1] == 1.2

    def test_numData(self):
        assert self.Data.numData == self.numPix**2

    def test_shift_coords(self):
        numPix = 10
        deltaPix = 0.05
        x_grid, y_grid, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix = util.make_grid_with_coordtransform(numPix=numPix, deltapix=deltaPix, subgrid_res=1, inverse=True)
        # mask (1= model this pixel, 0= leave blanck)

        kwargs_data = {'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
                       'transform_pix2angle': Mpix2coord, 'image_data': np.ones((numPix, numPix))}
        data = Data(kwargs_data)

        ra_shift = 0.05
        dec_shift = 0.
        kwargs_data['ra_shift'] = ra_shift
        kwargs_data['dec_shift'] = dec_shift
        data_shift = Data(kwargs_data)

        ra, dec = data.map_pix2coord(1, 1)
        ra_new, dec_new = data_shift.map_pix2coord(1, 1)
        npt.assert_almost_equal(ra_new - ra, ra_shift, decimal=10)
        npt.assert_almost_equal(dec_new - dec, dec_shift, decimal=10)

        ra_2, dec_2 = data_shift.map_pix2coord(2, 1)
        npt.assert_almost_equal(ra, ra_2, decimal=10)
        npt.assert_almost_equal(dec, dec_2, decimal=10)

        x, y = data.map_coord2pix(0, 0)
        x_new, y_new = data_shift.map_coord2pix(ra_shift, dec_shift)
        npt.assert_almost_equal(x, x_new, decimal=10)
        npt.assert_almost_equal(y, y_new, decimal=10)

    def test_shift_coordinate_grid(self):
        x_shift = 0.05
        y_shift = 0

        numPix = 10
        deltaPix = 0.05
        x_grid, y_grid, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix = util.make_grid_with_coordtransform(
            numPix=numPix, deltapix=deltaPix, subgrid_res=1, inverse=True)

        kwargs_data = {'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
                       'transform_pix2angle': Mpix2coord, 'image_data': np.ones((numPix, numPix))}

        data = Data(kwargs_data)
        data_new = copy.deepcopy(data)
        data_new.shift_coordinate_grid(x_shift, y_shift, pixel_unit=False)
        ra, dec = 0, 0
        x, y = data.map_coord2pix(ra, dec)
        x_new, y_new = data_new.map_coord2pix(ra + x_shift, dec + y_shift)
        npt.assert_almost_equal(x, x_new, decimal=10)
        npt.assert_almost_equal(y, y_new, decimal=10)

        ra, dec = data.map_pix2coord(x, y)
        ra_new, dec_new = data_new.map_pix2coord(x, y)
        npt.assert_almost_equal(ra, ra_new-x_shift, decimal=10)
        npt.assert_almost_equal(dec, dec_new-y_shift, decimal=10)

        x_coords, y_coords = data.coordinates
        x_coords_new, y_coords_new = data_new.coordinates
        npt.assert_almost_equal(x_coords[0], x_coords_new[0]-x_shift, decimal=10)
        npt.assert_almost_equal(y_coords[0], y_coords_new[0]-y_shift, decimal=10)


if __name__ == '__main__':
    pytest.main()
