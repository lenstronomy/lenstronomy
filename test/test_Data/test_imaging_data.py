import pytest
import numpy as np

import astrofunc.util as util
from lenstronomy.Data.imaging_data import Data


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
        idex_mask = util.image2array(idex_mask)
        image_data = np.zeros((5, 5))
        image_data[1, 1] = 1
        image_data = util.image2array(image_data)
        kwargs_data = {'idex_mask': idex_mask, 'image_data': image_data}
        data = Data(kwargs_data, subgrid_res=3)
        cut_data = data._cutout_psf(util.array2image(image_data), subgrid_res=1)
        print(cut_data)
        assert cut_data[0, 0] == 1
        assert cut_data[2, 1] == 0
        nx, ny = np.shape(cut_data)
        assert nx == 3
        assert ny == 2

        idex_mask = np.ones((5, 5))
        idex_mask = util.image2array(idex_mask)
        kwargs_data = {'idex_mask': idex_mask, 'image_data': image_data}
        data = Data(kwargs_data, subgrid_res=3)
        cut_data = data._cutout_psf(util.array2image(image_data), subgrid_res=1)
        assert cut_data[1, 1] == 1


if __name__ == '__main__':
    pytest.main()