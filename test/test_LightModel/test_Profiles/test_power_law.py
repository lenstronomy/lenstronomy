
import pytest
import numpy as np
import numpy.testing as npt
from lenstronomy.LightModel.Profiles.power_law import PowerLaw
from lenstronomy.LensModel.Profiles.spp import SPP
from lenstronomy.LensModel.Profiles.sis import SIS


class TestPowerLaw(object):
    """
    class to test the Moffat profile
    """
    def setup(self):
        pass

    def test_function(self):
        """

        :return:
        """
        profile = PowerLaw()
        spp = SPP()
        sis = SIS()
        x = np.linspace(0.1, 10, 10)
        kwargs_light = {'amp': 1., 'gamma': 2, 'e1': 0, 'e2': 0}
        kwargs_spp = {'theta_E': 1., 'gamma': 2}
        kwargs_sis = {'theta_E': 1.}
        flux = profile.function(x=x, y=1., **kwargs_light)
        f_xx, f_yy, f_xy = spp.hessian(x=x, y=1., **kwargs_spp)
        kappa_spp = 1/2. * (f_xx + f_yy)
        f_xx, f_yy, f_xy = sis.hessian(x=x, y=1., **kwargs_sis)
        kappa_sis = 1 / 2. * (f_xx + f_yy)
        npt.assert_almost_equal(kappa_sis, kappa_spp, decimal=5)
        npt.assert_almost_equal(flux/flux[0], kappa_sis/kappa_sis[0], decimal=5)


if __name__ == '__main__':
    pytest.main()
