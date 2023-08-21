
import pytest
import numpy as np
import numpy.testing as npt
from lenstronomy.LightModel.Profiles.power_law import PowerLaw
from lenstronomy.LensModel.Profiles.spp import SPP
from lenstronomy.LensModel.Profiles.sis import SIS
from lenstronomy.LensModel.Profiles.nie import NIE
from lenstronomy.LensModel.Profiles.epl import EPL


class TestPowerLaw(object):
    """Class to test the Moffat profile."""
    def setup_method(self):
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
        f_xx, f_xy, f_yx, f_yy = spp.hessian(x=x, y=1., **kwargs_spp)
        kappa_spp = 1/2. * (f_xx + f_yy)
        f_xx, f_xy, f_yx, f_yy = sis.hessian(x=x, y=1., **kwargs_sis)
        kappa_sis = 1 / 2. * (f_xx + f_yy)
        npt.assert_almost_equal(kappa_sis, kappa_spp, decimal=5)
        npt.assert_almost_equal(flux/flux[0], kappa_sis/kappa_sis[0], decimal=5)

        # test against NIE
        nie = NIE()
        e1, e2 = 0.2, -0.1
        kwargs_light = {'amp': 1., 'gamma': 2, 'e1': e1, 'e2': e2}
        kwargs_nie = {'theta_E': 1., 'e1': e1, 'e2': e2, 's_scale': 0.00001}
        flux = profile.function(x=x, y=1., **kwargs_light)
        f_xx, f_xy, f_yx, f_yy = nie.hessian(x=x, y=1., **kwargs_nie)
        kappa_nie = 1/2. * (f_xx + f_yy)
        npt.assert_almost_equal(flux/flux[0], kappa_nie/kappa_nie[0], decimal=5)

        # test against EPL
        epl = EPL()
        e1, e2 = 0.2, -0.1
        kwargs_light = {'amp': 1., 'gamma': 2, 'e1': e1, 'e2': e2}
        kwargs_epl = {'theta_E': 1., 'e1': e1, 'e2': e2, 'gamma': 2}
        flux = profile.function(x=x, y=1., **kwargs_light)
        f_xx, f_xy, f_yx, f_yy = epl.hessian(x=x, y=1., **kwargs_epl)
        kappa_epl = 1 / 2. * (f_xx + f_yy)
        npt.assert_almost_equal(flux / flux[0], kappa_epl / kappa_epl[0], decimal=5)


if __name__ == '__main__':
    pytest.main()
