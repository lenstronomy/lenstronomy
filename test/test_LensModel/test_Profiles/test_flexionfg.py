__author__ = 'sibirrer'


from lenstronomy.LensModel.Profiles.flexionfg import Flexionfg
import numpy.testing as npt
import pytest


class TestFlexionfg(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.flex = Flexionfg()
        F1, F2, G1, G2= 0.01, 0.02, 0.03, 0.04
        g1 = (3 * F1 + G1) * 0.5
        g2 = (3 * F2 + G2) * 0.5
        g3 = (F1 - G1) * 0.5
        g4 = (F2 - G2) * 0.5
        self.kwargs_lens = {'F1': F1, 'F2': F2, 'G1': G1, 'G2': G2}

    def test_transform_fg(self):
        values=self.flex.transform_fg(**self.kwargs_lens)
        F1, F2, G1, G2 = 0.01, 0.02, 0.03, 0.04
        g1 = (3 * F1 + G1) * 0.5
        g2 = (3 * F2 + G2) * 0.5
        g3 = (F1 - G1) * 0.5
        g4 = (F2 - G2) * 0.5
        npt.assert_almost_equal(values[0], g1, decimal=5)
        npt.assert_almost_equal(values[1], g2, decimal=5)
        npt.assert_almost_equal(values[2], g3, decimal=5)
        npt.assert_almost_equal(values[3], g4, decimal=5)


if __name__ == '__main__':
    pytest.main()