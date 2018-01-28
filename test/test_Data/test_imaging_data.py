import pytest
import numpy as np
import numpy.testing as npt

from lenstronomy.Data.imaging_data import Data
import lenstronomy.Util.util as util


class TestData(object):
    def setup(self):
        kwargs_data = {}
        self.Data = Data(kwargs_data)

    def test_idex_subgrid(self):
        idex_mask = np.zeros(100)
        n = 8
        nx, ny = int(np.sqrt(len(idex_mask))), int(np.sqrt(len(idex_mask)))
        idex_mask[n] = 1
        subgrid_res = 2
        idex_mask_subgrid = self.Data._subgrid_idex(idex_mask, subgrid_res, nx, ny)
        assert idex_mask_subgrid[(n + 1) * subgrid_res - 1] == 1
        assert idex_mask_subgrid[(n + 1) * subgrid_res - 2] == 1
        print(type(nx * subgrid_res + (n + 1) * subgrid_res - 1))
        print(type((n + 1) * subgrid_res - 2))
        assert idex_mask_subgrid[nx * subgrid_res + (n + 1) * subgrid_res - 1] == 1

    def test_get_covariance_matrix(self):
        d = np.array([1, 2, 3])
        sigma_b = 1
        f = 10.
        result = self.Data.covariance_matrix(d, sigma_b, f)
        assert result[0] == 1.1
        assert result[1] == 1.2

    def test_psf_cutout(self):
        idex_mask = np.zeros((5, 5))
        idex_mask[3, 2] = 1
        idex_mask[1, 1] = 1
        image_data = np.zeros((5, 5))
        image_data[1, 1] = 1
        kwargs_data = {'idex_mask': idex_mask, 'image_data': image_data}
        data = Data(kwargs_data, subgrid_res=3)
        cut_data = data._cutout_psf(image_data, subgrid_res=1)
        print(cut_data)
        assert cut_data[0, 0] == 1
        assert cut_data[2, 1] == 0
        nx, ny = np.shape(cut_data)
        assert nx == 3
        assert ny == 2

        idex_mask = np.ones((5, 5))
        kwargs_data = {'idex_mask': idex_mask, 'image_data': image_data}
        data = Data(kwargs_data, subgrid_res=3)
        cut_data = data._cutout_psf(image_data, subgrid_res=1)
        assert cut_data[1, 1] == 1

    def test_shift_coords(self):
        numPix = 10
        deltaPix = 0.05
        x_grid, y_grid, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix = util.make_grid_with_coordtransform(numPix=numPix, deltapix=deltaPix, subgrid_res=1)
        # mask (1= model this pixel, 0= leave blanck)

        kwargs_data = {'x_coords': x_grid, 'y_coords': y_grid, 'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
                       'transform_pix2angle': Mpix2coord, 'image_data': np.ones((numPix, numPix))}
        data = Data(kwargs_data, subgrid_res=3)

        ra_shift = 0.05
        dec_shift = 0.
        kwargs_data['ra_shift'] = ra_shift
        kwargs_data['dec_shift'] = dec_shift
        data_shift = Data(kwargs_data, subgrid_res=3)

        ra, dec = data.map_pix2coord(1, 1)
        ra_new, dec_new = data_shift.map_pix2coord(1, 1)
        npt.assert_almost_equal(ra_new - ra, ra_shift, decimal=10)
        npt.assert_almost_equal(dec_new - dec, dec_shift, decimal=10)

        ra_2, dec_2 = data_shift.map_pix2coord(0, 1)
        npt.assert_almost_equal(ra, ra_2, decimal=10)
        npt.assert_almost_equal(dec, dec_2, decimal=10)

        x, y = data.map_coord2pix(0, 0)
        x_new, y_new = data_shift.map_coord2pix(ra_shift, dec_shift)
        npt.assert_almost_equal(x, x_new, decimal=10)
        npt.assert_almost_equal(y, y_new, decimal=10)



if __name__ == '__main__':
    pytest.main()