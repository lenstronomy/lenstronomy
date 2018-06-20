
import pytest
import numpy as np
import numpy.testing as npt
from lenstronomy.LightModel.Profiles.power_law import PowerLaw
from lenstronomy.LightModel.Profiles.chameleon import Chameleon


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
        chameleon = Chameleon()
        profile = PowerLaw()

        x = np.linspace(0.1, 10, 10)
        kwargs_light = {'amp': 1., 'flux_ratio': -1., 'gamma1': 2., 'gamma2': 1., 'e1': 0, 'e2': 0}
        kwargs_1 = {'amp': 1., 'gamma': 2., 'e1': 0, 'e2': 0}
        kwargs_2 = {'amp': -1., 'gamma': 1., 'e1': 0, 'e2': 0}
        flux = chameleon.function(x=x, y=1., **kwargs_light)
        flux1 = profile.function(x=x, y=1., **kwargs_1)
        flux2 = profile.function(x=x, y=1., **kwargs_2)
        npt.assert_almost_equal(flux, flux1 + flux2, decimal=5)


if __name__ == '__main__':
    pytest.main()
