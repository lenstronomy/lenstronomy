from lenstronomy.LightModel.Profiles.linear import Linear, LinearEllipse
import numpy as np
import numpy.testing as npt

class TestLinear(object):

    def setup_method(self):
        self.linear = Linear()
        self.ellipse = LinearEllipse()

    def test_function(self):

        amp = 1
        k = 1
        x = np.array([0, 1])
        y = np.array([0, 0])
        flux_true = np.array([amp, amp+k])
        flux = self.linear.function(x, y, amp=amp, k=k)
        flux_ellipse = self.ellipse.function(x, y, amp=amp, k=k, e1=0, e2=0)
        npt.assert_almost_equal(flux_ellipse, flux, decimal=6)
        npt.assert_almost_equal(flux, flux_true, decimal=5)

    def test_total_flux(self):
        amp = 1
        k = 1
        total_flux = self.linear.total_flux(amp, k, center_x=0, center_y=0)
        total_flux_ellipse = self.ellipse.total_flux(amp, k, center_x=0, center_y=0)
        npt.assert_almost_equal(total_flux_ellipse, total_flux, decimal=6)