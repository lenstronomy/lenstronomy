import numpy as np
from lenstronomy.LensModel.Profiles.sersic_utils import SersicUtil
import numpy.testing as npt


def test_bn():
    n_array = np.linspace(start=0.2, stop=8, num=30)
    for n in n_array:
        bn_approx = 1.9992 * n - 0.3271
        bn = SersicUtil.b_n(n)
        npt.assert_almost_equal(bn, bn_approx, decimal=3)
