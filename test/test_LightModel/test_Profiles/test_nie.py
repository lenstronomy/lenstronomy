
import pytest
import numpy as np
import numpy.testing as npt
from lenstronomy.LightModel.Profiles.nie import NIE as NIE_light
from lenstronomy.LensModel.Profiles.nie import NIE as NIE_lens


class TestNIE(object):
    """
    class to test the Moffat profile
    """
    def setup(self):
        pass

    def test_function(self):
        """

        :return:
        """
        lens = NIE_lens()
        light = NIE_light()

        x = np.linspace(0.1, 10, 10)
        e1, e2 = 0.1, 0
        s = 0.2
        kwargs_light = {'amp': 1., 'e1': e1, 'e2': e2, 's_scale': s}
        kwargs_lens = {'theta_E': 1., 'e1': e1, 'e2': e2, 's_scale': s}
        flux = light.function(x=x, y=1., **kwargs_light)
        f_xx, f_yy, f_xy = lens.hessian(x=x, y=1., **kwargs_lens)
        kappa = 1/2. * (f_xx + f_yy)
        npt.assert_almost_equal(flux/flux[-1], kappa/kappa[-1], decimal=4)


if __name__ == '__main__':
    pytest.main()
