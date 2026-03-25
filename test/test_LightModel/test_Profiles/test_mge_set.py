import numpy as np

from lenstronomy.LightModel.Profiles.mge_set import MGESet
from lenstronomy.LightModel.Profiles.gaussian import MultiGaussian
import numpy.testing as npt


class TestMGESet(object):
    """Class to test the Gaussian profile."""

    def setup_method(self):
        self.n_comp = 20
        self.multi_gauss = MultiGaussian()
        self.mge_set = MGESet(n_comp=self.n_comp)

    def test_function(self):
        x, y = 3, 2
        sigma_min, sigma_width = 0.01, 10
        sigma = np.logspace(
            start=np.log10(sigma_min),
            stop=np.log10(sigma_min + sigma_width),
            num=self.n_comp,
        )
        amp = np.ones(self.n_comp)
        flux_mg = self.multi_gauss.function(
            x, y, amp=amp, sigma=sigma, center_x=0, center_y=0
        )
        flux_mge = self.mge_set.function(
            x,
            y,
            amp=amp,
            sigma_min=sigma_min,
            sigma_width=sigma_width,
            center_x=0,
            center_y=0,
        )
        npt.assert_almost_equal(flux_mge, flux_mg, decimal=8)

    def test_function_split(self):
        """

        :return:
        """
        x, y = np.linspace(start=0, stop=10, num=10), np.zeros(10)
        sigma_min, sigma_width = 0.01, 10
        sigma = np.logspace(
            start=np.log10(sigma_min),
            stop=np.log10(sigma_min + sigma_width),
            num=self.n_comp,
        )
        amp = np.ones(self.n_comp)
        flux_mg = self.multi_gauss.function_split(
            x, y, amp=amp, sigma=sigma, center_x=0, center_y=0
        )
        flux_mge = self.mge_set.function_split(
            x,
            y,
            amp=amp,
            sigma_min=sigma_min,
            sigma_width=sigma_width,
            center_x=0,
            center_y=0,
        )
        npt.assert_almost_equal(flux_mge[0], flux_mg[0], decimal=8)
        npt.assert_almost_equal(flux_mge[1], flux_mg[1], decimal=8)

    def test_light3d(self):
        r = np.linspace(start=0, stop=10, num=10)
        sigma_min, sigma_width = 0.01, 10
        sigma = np.logspace(
            start=np.log10(sigma_min),
            stop=np.log10(sigma_min + sigma_width),
            num=self.n_comp,
        )
        amp = np.ones(self.n_comp)
        light_3d = self.multi_gauss.light_3d(r, amp=amp, sigma=sigma)
        light_3d_mge = self.mge_set.light_3d(
            r, amp=amp, sigma_min=sigma_min, sigma_width=sigma_width
        )
        npt.assert_almost_equal(light_3d_mge, light_3d, decimal=8)
        npt.assert_almost_equal(light_3d_mge, light_3d, decimal=8)

    def test_total_flux(self):
        r = np.linspace(start=0, stop=10, num=10)
        sigma_min, sigma_max = 0.01, 10
        sigma = np.logspace(
            start=np.log10(sigma_min), stop=np.log10(sigma_max), num=self.n_comp
        )
        amp = np.ones(self.n_comp)
        flux = self.multi_gauss.total_flux(amp=amp, sigma=sigma)
        flux_mge = self.mge_set.total_flux(
            amp=amp, sigma_min=sigma_min, sigma_width=sigma_max
        )
        npt.assert_almost_equal(flux_mge, flux, decimal=8)
        npt.assert_almost_equal(flux_mge, flux, decimal=8)

    def test_num_linear(self):
        num = self.mge_set.num_linear
        assert num == self.n_comp
