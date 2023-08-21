import pytest
import numpy as np
import numpy.testing as npt
from lenstronomy.Analysis import light2mass
from lenstronomy.LightModel.light_model import LightModel


class TestLight2Mass(object):

    def setup_method(self):
        pass

    def test_light2mass_conversion(self):
        numPix = 100
        deltaPix = 0.05

        lightModel = LightModel(light_model_list=['SERSIC_ELLIPSE', 'SERSIC'])
        kwargs_lens_light = [{'R_sersic': 0.5, 'n_sersic': 4, 'amp': 2, 'e1': 0, 'e2': 0.05},
                             {'R_sersic': 1.5, 'n_sersic': 1, 'amp': 2}]

        kwargs_interpol = light2mass.light2mass_interpol(lens_light_model_list=['SERSIC_ELLIPSE', 'SERSIC'],
                                                           kwargs_lens_light=kwargs_lens_light, numPix=numPix,
                                                           deltaPix=deltaPix, subgrid_res=1)
        from lenstronomy.LensModel.lens_model import LensModel
        lensModel = LensModel(lens_model_list=['INTERPOL_SCALED'])
        kwargs_lens = [kwargs_interpol]
        import lenstronomy.Util.util as util
        x_grid, y_grid = util.make_grid(numPix, deltapix=deltaPix)
        kappa = lensModel.kappa(x_grid, y_grid, kwargs=kwargs_lens)
        kappa = util.array2image(kappa)
        kappa /= np.mean(kappa)
        flux = lightModel.surface_brightness(x_grid, y_grid, kwargs_lens_light)
        flux = util.array2image(flux)
        flux /= np.mean(flux)
        # import matplotlib.pyplot as plt
        # plt.matshow(flux-kappa)
        # plt.colorbar()
        # plt.show()
        delta_kappa = (kappa - flux) / flux
        max_delta = np.max(np.abs(delta_kappa))
        assert max_delta < 1
        # assert max_diff < 0.01
        npt.assert_almost_equal(flux[0, 0], kappa[0, 0], decimal=2)


if __name__ == '__main__':
    pytest.main()
