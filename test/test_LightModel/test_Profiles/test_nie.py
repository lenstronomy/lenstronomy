
import pytest
import numpy.testing as npt
import numpy as np
from lenstronomy.LightModel.Profiles.nie import NIE as NIE_light
from lenstronomy.LensModel.Profiles.nie import NIE as NIE_lens
from lenstronomy.Util import util


class TestNIE(object):
    """
    class to test the NIE profile
    """
    def setup_method(self):
        pass

    def test_function(self):
        """

        :return:
        """
        lens = NIE_lens()
        light = NIE_light()

        x, y = util.make_grid(numPix=100, deltapix=0.1)
        e1, e2 = 0.2, 0
        s = 1.
        kwargs_light = {'amp': 1., 'e1': e1, 'e2': e2, 's_scale': s}
        kwargs_lens = {'theta_E': 1., 'e1': e1, 'e2': e2, 's_scale': s}
        flux = light.function(x=x, y=y, **kwargs_light)
        f_xx, f_xy, f_yx, f_yy = lens.hessian(x=x, y=y, **kwargs_lens)
        kappa = 1/2. * (f_xx + f_yy)

        npt.assert_almost_equal(flux/flux[-1], kappa/kappa[-1], decimal=3)

        # test whether ellipticity changes overall flux normalization
        kwargs_light_round = {'amp': 1., 'e1': 0, 'e2': 0, 's_scale': s}

        x_, y_ = util.points_on_circle(radius=1, num_points=20)
        f_r = light.function(x=x_, y=y_, **kwargs_light)
        f_r = np.mean(f_r)

        f_r_round = light.function(x=1, y=0, **kwargs_light_round)
        npt.assert_almost_equal(f_r / f_r_round, 1, decimal=2)


if __name__ == '__main__':
    pytest.main()
