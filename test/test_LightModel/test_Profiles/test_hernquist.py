from lenstronomy.LightModel.Profiles.hernquist import Hernquist, HernquistEllipse
from lenstronomy.Util.util import make_grid
import numpy as np
import numpy.testing as npt


class TestHernquist(object):

    def setup_method(self):
        self.hernquist = Hernquist()
        self.hernquist_ellipse = HernquistEllipse()

    def test_total_flux(self):
        delta_pix = 0.2
        x, y = make_grid(numPix=1000, deltapix=delta_pix)

        rs, amp = 1, 1
        total_flux = self.hernquist.total_flux(amp=amp, Rs=rs)
        flux = self.hernquist.function(x, y, amp=amp, Rs=rs)
        total_flux_numerics = np.sum(flux) * delta_pix**2
        npt.assert_almost_equal(total_flux_numerics / total_flux, 1, decimal=1)

        total_flux_ellipse = self.hernquist_ellipse.total_flux(amp=amp, Rs=rs)
        npt.assert_almost_equal(total_flux_ellipse, total_flux)
