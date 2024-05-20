from lenstronomy.LightModel.Profiles.lineprofile import LineProfile
import numpy as np
import numpy.testing as npt


class TestLineProfile(object):
    def setup_method(self):
        self.lineprofile = LineProfile()

    def test_function(self):
        amp = 1
        length = 1
        width = 0.01
        angle = 57
        x = np.array([0, 1, 0.5 * np.cos(np.deg2rad(angle))])
        y = np.array([0, 0, 0.5 * np.sin(np.deg2rad(angle))])
        x_single = 0.2 * np.cos(np.deg2rad(angle))
        y_single = 0.2 * np.sin(np.deg2rad(angle))
        flux_true = np.array([amp, 0, amp])
        flux = self.lineprofile.function(x, y, amp, angle, length, width)
        single_flux_true = amp
        single_flux = self.lineprofile.function(x_single, y_single, amp, angle, length, width)
        npt.assert_equal(flux_true, flux)
        npt.assert_equal(single_flux_true, single_flux)

    def test_total_flux(self):
        amp = 1
        length = 1
        width = 0.01
        angle = 57
        total_flux = self.lineprofile.total_flux(amp, angle, length, width)
        total_flux_true = length * width * amp
        npt.assert_equal(total_flux_true, total_flux)
