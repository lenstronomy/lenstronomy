import pytest
import numpy as np

from lenstronomy.ImSim.data import Data


class TestData(object):
    def setup(self):
        kwargs_options = {'system_name': '', 'data_file': ''
            , 'cosmo_file': '', 'lens_model_list': ['GAUSSIAN'], 'source_light_model_list': ['GAUSSIAN'], 'lens_light_model_list': ['SERSIC']
            , 'subgrid_res': 10, 'numPix': 200, 'psf_type': 'gaussian', 'x2_simple': True}
        kwargs_data = {}
        self.Data = Data(kwargs_options, kwargs_data)

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

if __name__ == '__main__':
    pytest.main()