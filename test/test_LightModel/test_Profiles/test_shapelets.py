import lenstronomy.Util.util as util
import numpy as np
import numpy.testing as npt
import pytest
import unittest
from lenstronomy.LightModel.Profiles.shapelets import Shapelets, ShapeletSet


class TestShapeletSet(object):
    """
    class to test Shapelets
    """
    def setup(self):
        self.shapeletSet = ShapeletSet()
        self.shapelets = Shapelets(precalc=False)
        self.x, self.y = util.make_grid(10, 0.1, 1)

    def test_shapelet_set(self):
        """

        :return:
        """
        n_max = 2
        beta = 1.
        amp = [1,0,0,0,0,0]
        output = self.shapeletSet.function(np.array(1), np.array(1), amp, n_max, beta, center_x=0, center_y=0)
        assert output == 0.20755374871029739
        input = np.array(0.)
        input += output

        output = self.shapeletSet.function(self.x, self.y, amp, n_max, beta, center_x=0, center_y=0)
        assert output[10] == 0.47957022395315946
        output = self.shapeletSet.function(1, 1, amp, n_max, beta, center_x=0, center_y=0)
        assert output == 0.20755374871029739

        n_max = -1
        beta = 1.
        amp = [1, 0, 0, 0, 0, 0]
        output = self.shapeletSet.function(np.array(1), np.array(1), amp, n_max, beta, center_x=0, center_y=0)
        assert output == 0

        beta = 1.
        amp = 1
        shapelets = Shapelets(precalc=False, stable_cut=False)
        output = shapelets.function(np.array(1), np.array(1), amp, beta, 0, 0, center_x=0, center_y=0)
        npt.assert_almost_equal(0.2075537487102974 , output, decimal=8)

    def test_shapelet_basis(self):
        num_order = 5
        beta = 1
        numPix = 10
        kernel_list = self.shapeletSet.shapelet_basis_2d(num_order, beta, numPix)
        assert kernel_list[0][4, 4] == 0.43939128946772255

    def test_decomposition(self):
        """

        :return:
        """
        n_max = 2
        beta = 10.
        deltaPix = 1
        amp = np.array([1,1,1,1,1,1])
        x, y = util.make_grid(100, deltaPix, 1)
        input = self.shapeletSet.function(x, y, amp, n_max, beta, center_x=0, center_y=0)
        amp_out = self.shapeletSet.decomposition(input, x, y, n_max, beta, deltaPix, center_x=0, center_y=0)
        for i in range(len(amp)):
            npt.assert_almost_equal(amp_out[i], amp[i], decimal=4)

    def test_function_split(self):
        n_max = 2
        beta = 10.
        deltaPix = 0.1
        amp = np.array([1,1,1,1,1,1])
        x, y = util.make_grid(10, deltaPix, 1)
        function_set = self.shapeletSet.function_split(x, y, amp, n_max, beta, center_x=0, center_y=0)
        test_flux = self.shapelets.function(x, y, amp=1., n1=0, n2=0, beta=beta, center_x=0, center_y=0)
        print(np.shape(function_set))
        print(np.shape(test_flux))
        assert function_set[0][10] == test_flux[10]

    def test_interpolate(self):
        shapeletsInterp = Shapelets(interpolation=True)
        x, y = 0.99, 0
        beta = 0.5
        flux_full = self.shapelets.function(x, y, amp=1., n1=0, n2=0, beta=beta, center_x=0, center_y=0)
        flux_interp = shapeletsInterp.function(x, y, amp=1., n1=0, n2=0, beta=beta, center_x=0, center_y=0)
        npt.assert_almost_equal(flux_interp, flux_full, decimal=10)

    def test_hermval(self):
        x = np.linspace(0, 2000, 2001)
        n_array = [1, 2, 3, 0, 1]
        import numpy.polynomial.hermite as hermite
        out_true = hermite.hermval(x, n_array)
        out_approx = self.shapelets.hermval(x, n_array)
        shape_true = out_true * np.exp(-x ** 2 / 2.)
        shape_approx = out_approx * np.exp(-x ** 2 / 2.)
        npt.assert_almost_equal(shape_approx, shape_true, decimal=6)

        x = 2
        n_array = [1, 2, 3, 0, 1]
        out_true = hermite.hermval(x, n_array)
        out_approx = self.shapelets.hermval(x, n_array)
        npt.assert_almost_equal(out_approx, out_true, decimal=6)

        x = 2001
        n_array = [1, 2, 3, 0, 1]
        out_true = hermite.hermval(x, n_array)
        out_approx = self.shapelets.hermval(x, n_array)
        shape_true = out_true * np.exp(-x**2/2.)
        shape_approx = out_approx * np.exp(-x ** 2 / 2.)
        npt.assert_almost_equal(shape_approx, shape_true, decimal=6)


class TestRaise(unittest.TestCase):

    def test_raise(self):
        with self.assertRaises(ValueError):
            shapelets = Shapelets()
            shapelets.pre_calc(1, 1, beta=1, n_order=200, center_x=0, center_y=0)


if __name__ == '__main__':
    pytest.main()
