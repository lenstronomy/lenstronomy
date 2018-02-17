__author__ = 'sibirrer'


from lenstronomy.LightModel.Profiles.sersic import Sersic, Sersic_elliptic, DoubleSersic, CoreSersic, DoubleCoreSersic, BuldgeDisk

import numpy as np
import pytest
import numpy.testing as npt


class TestSersic(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.sersic = Sersic(smoothing=0.02)
        self.sersic_elliptic = Sersic_elliptic(smoothing=0.02)
        self.double_sersic = DoubleSersic(smoothing=0.02)
        self.core_sersic = CoreSersic(smoothing=0.02)
        self.double_core_sersic = DoubleCoreSersic(smoothing=0.02)
        self.buldge_disk = BuldgeDisk(smoothing=0.02)

    def test_sersic(self):
        x = np.array([1])
        y = np.array([2])
        I0_sersic = 1
        R_sersic = 1
        n_sersic = 1
        center_x = 0
        center_y = 0
        values = self.sersic.function(x, y, I0_sersic, R_sersic, n_sersic, center_x, center_y)
        npt.assert_almost_equal(values[0], 0.12658651833626802, decimal=6)
        x = np.array([0])
        y = np.array([0])
        values = self.sersic.function( x, y, I0_sersic, R_sersic, n_sersic, center_x, center_y)
        npt.assert_almost_equal(values[0],  5.1482559148107292, decimal=2)

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.sersic.function( x, y, I0_sersic, R_sersic, n_sersic, center_x, center_y)
        npt.assert_almost_equal(values[0], 0.12658651833626802, decimal=6)
        npt.assert_almost_equal(values[1], 0.026902273598180083, decimal=6)
        npt.assert_almost_equal(values[2], 0.0053957432862338055, decimal=6)

    def test_symmetry_r_sersic(self):
        x = np.array([2,3,4])
        y = np.array([1,1,1])
        I0_sersic = 1
        R_sersic1 = 1
        R_sersic2 = 0.1
        n_sersic = 1
        center_x = 0
        center_y = 0
        values1 = self.sersic.function(x*R_sersic1, y*R_sersic1, I0_sersic, R_sersic1, n_sersic, center_x, center_y)
        values2 = self.sersic.function(x*R_sersic2, y*R_sersic2, I0_sersic, R_sersic2, n_sersic, center_x, center_y)
        npt.assert_almost_equal(values1[0], values2[0], decimal=6)
        npt.assert_almost_equal(values1[1], values2[1], decimal=6)
        npt.assert_almost_equal(values1[2], values2[2], decimal=6)

    def test_sersic_center(self):
        x = 0.01
        y = 0.
        I0_sersic = 1
        R_sersic = 0.1
        n_sersic = 4.
        center_x = 0
        center_y = 0
        values = self.sersic.function(x, y, I0_sersic, R_sersic, n_sersic, center_x, center_y)
        npt.assert_almost_equal(values, 12.688073819377406, decimal=6)

    def test_sersic_elliptic(self):
        x = np.array([1])
        y = np.array([2])
        I0_sersic = 1
        R_sersic = 1
        n_sersic = 1
        phi_G = 1
        q = 0.9
        center_x = 0
        center_y = 0
        values = self.sersic_elliptic.function(x, y, I0_sersic, R_sersic, n_sersic, phi_G, q, center_x, center_y)
        npt.assert_almost_equal(values[0], 0.12595366113005077, decimal=6)
        x = np.array([0])
        y = np.array([0])
        values = self.sersic_elliptic.function(x, y, I0_sersic, R_sersic, n_sersic, phi_G, q, center_x, center_y)
        npt.assert_almost_equal(values[0], 5.1482553482055664, decimal=2)

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.sersic_elliptic.function(x, y, I0_sersic, R_sersic, n_sersic, phi_G, q, center_x, center_y)
        npt.assert_almost_equal(values[0], 0.11308277793465012, decimal=6)
        npt.assert_almost_equal(values[1], 0.021188620675507107, decimal=6)
        npt.assert_almost_equal(values[2], 0.0037276744362724477, decimal=6)

    def test_core_sersic(self):
        x = np.array([1])
        y = np.array([2])
        I0 = 1
        Rb = 1
        Re = 2
        gamma = 3
        n = 1
        phi_G = 1
        q = 0.9
        center_x = 0
        center_y = 0
        values = self.core_sersic.function(x, y, I0, Rb, Re, n, gamma, phi_G, q, center_x, center_y)
        npt.assert_almost_equal(values[0], 0.84489101, decimal=8)
        x = np.array([0])
        y = np.array([0])
        values = self.core_sersic.function(x, y, I0, Rb, Re, n, gamma, phi_G, q, center_x, center_y)
        npt.assert_almost_equal(values[0], 288406.09, decimal=0)

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.core_sersic.function(x, y, I0, Rb, Re, n, gamma, phi_G, q, center_x, center_y)
        npt.assert_almost_equal(values[0], 0.79749529635325933, decimal=6)
        npt.assert_almost_equal(values[1], 0.33653478121594838, decimal=6)
        npt.assert_almost_equal(values[2], 0.14050402887681532, decimal=6)

    def test_double_sersic(self):
        x = np.array([1])
        y = np.array([2])
        I0_sersic = 1
        R_sersic = 1
        n_sersic = 1
        phi_G = 1
        q = 0.9
        I0_2 = 0.1
        R_2 = 2
        n_2 = 2
        phi_G_2 = 1
        q_2 = 1
        center_x = 0
        center_y = 0
        values = self.double_sersic.function(x, y, I0_sersic, R_sersic, n_sersic, phi_G, q, I0_2, R_2, n_2, phi_G_2, q_2, center_x, center_y)
        npt.assert_almost_equal(values[0], 0.20696126663199443, decimal=8)
        x = np.array([0])
        y = np.array([0])
        values = self.double_sersic.function(x, y, I0_sersic, R_sersic, n_sersic, phi_G, q, I0_2, R_2, n_2, phi_G_2, q_2, center_x, center_y)
        npt.assert_almost_equal(values[0], 7.8708484172821045, decimal=8)

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.double_sersic.function(x, y, I0_sersic, R_sersic, n_sersic, phi_G, q, I0_2, R_2, n_2, phi_G_2, q_2, center_x, center_y)
        npt.assert_almost_equal(values[0], 0.19409037374964733, decimal=8)
        npt.assert_almost_equal(values[1], 0.060052096255106595, decimal=8)
        npt.assert_almost_equal(values[2], 0.023917479151437715, decimal=8)

    def test_double_sersic_function_split(self):
        x = np.array([1])
        y = np.array([2])
        I0_sersic = 1
        R_sersic = 1
        n_sersic = 1
        phi_G = 1
        q = 0.9
        I0_2 = 0.1
        R_2 = 2
        n_2 = 2
        phi_G_2 = 1
        q_2 = 1
        center_x = 0
        center_y = 0
        func_1, func_2 = self.double_sersic.function_split(x, y, I0_sersic, R_sersic, n_sersic, phi_G, q, I0_2, R_2, n_2, phi_G_2, q_2, center_x, center_y)
        npt.assert_almost_equal(func_1[0], 0.12595365941524506)
        npt.assert_almost_equal(func_2[0], 0.081007614731788635)

    def test_double_core_sersic(self):
        x = np.array([1])
        y = np.array([2])
        I0_sersic = 1
        R_sersic = 1
        n_sersic = 1
        phi_G = 1
        q = 0.9
        I0_2 = 0.1
        R_2 = 2
        n_2 = 2
        Re = 0.1
        gamma = 2
        values = self.double_core_sersic.function(x, y, I0_sersic, Re, R_sersic, n_sersic, gamma, phi_G, q, I0_2, R_2, n_2, phi_G_2=0, q_2=1, center_x=0, center_y=0)
        npt.assert_almost_equal(values[0], 0.20695476233959198, decimal=5)
        x = np.array([0])
        y = np.array([0])
        values = self.double_core_sersic.function(x, y, I0_sersic, Re, R_sersic, n_sersic, gamma, phi_G, q, I0_2, R_2, n_2, phi_G_2=0, q_2=1, center_x=0, center_y=0)
        npt.assert_almost_equal(values[0], 115.86341203723811, decimal=5)

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.double_core_sersic.function(x, y, I0_sersic, Re, R_sersic, n_sersic, gamma, phi_G, q, I0_2, R_2, n_2, phi_G_2=0, q_2=1, center_x=0, center_y=0)
        npt.assert_almost_equal(values[0], 0.19408468241596671, decimal=5)
        npt.assert_almost_equal(values[1], 0.060051398351788521, decimal=5)
        npt.assert_almost_equal(values[2], 0.023917404236271977, decimal=5)

    def test_double_core_sersic_function_split(self):
        x = np.array([1])
        y = np.array([2])
        I0_sersic = 1
        R_sersic = 1
        n_sersic = 1
        phi_G = 1
        q = 0.9
        I0_2 = 0.1
        R_2 = 2
        n_2 = 2
        Re = 0.1
        gamma = 2
        func_1, func_2 = self.double_core_sersic.function_split(x, y, I0_sersic, Re, R_sersic, n_sersic, gamma, phi_G, q, I0_2, R_2, n_2, phi_G_2=0, q_2=1, center_x=0, center_y=0)
        npt.assert_almost_equal(func_1[0], 0.12594715)
        npt.assert_almost_equal(func_2[0], 0.081007614731788635)

    def test_buldge_disk(self):
        x = np.array([1])
        y = np.array([2])
        I0_b = 1
        R_b = 1
        phi_G_b = 0
        q_b = 0.9
        I0_d = 1
        R_d = 2
        phi_G_d = 0
        q_d = 0.7

        values = self.buldge_disk.function(x, y, I0_b, R_b, phi_G_b, q_b, I0_d, R_d, phi_G_d, q_d, center_x=0, center_y=0)
        npt.assert_almost_equal(values[0], 0.57135696709156036, decimal=8)
        x = np.array([0])
        y = np.array([0])
        values = self.buldge_disk.function(x, y, I0_b, R_b, phi_G_b, q_b, I0_d, R_d, phi_G_d, q_d, center_x=0, center_y=0)
        npt.assert_almost_equal(values[0], 124.98796224594116, decimal=8)

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.buldge_disk.function(x, y, I0_b, R_b, phi_G_b, q_b, I0_d, R_d, phi_G_d, q_d, center_x=0, center_y=0)
        npt.assert_almost_equal(values[0], 0.85350380837917328, decimal=8)
        npt.assert_almost_equal(values[1], 0.40610484033823013, decimal=8)
        npt.assert_almost_equal(values[2], 0.19044427201151848, decimal=8)

    def test_buldge_disk_function_split(self):
        x = np.array([1])
        y = np.array([2])
        I0_b = 1
        R_b = 1
        phi_G_b = 0
        q_b = 0.9
        I0_d = 1
        R_d = 2
        phi_G_d = 0
        q_d = 0.7
        func_1, func_2 = self.buldge_disk.function_split(x, y, I0_b, R_b, phi_G_b, q_b, I0_d, R_d, phi_G_d, q_d, center_x=0, center_y=0)
        npt.assert_almost_equal(func_1[0], 0.1476433128118515)
        npt.assert_almost_equal(func_2[0], 0.42371365427970886)



if __name__ == '__main__':
    pytest.main()