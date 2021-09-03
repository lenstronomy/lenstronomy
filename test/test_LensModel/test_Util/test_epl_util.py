from lenstronomy.LensModel.Util.epl_util import brentq_nojit
import numpy as np
import numpy.testing as npt
def test_brentq_nojit():
    npt.assert_almost_equal(brentq_nojit(lambda x, args: np.sin(x), np.pi/2, 3*np.pi/2), np.pi, decimal=10)
    npt.assert_almost_equal(brentq_nojit(lambda x, args: np.cos(x), np.pi, np.pi/2), np.pi/2, decimal=10)
