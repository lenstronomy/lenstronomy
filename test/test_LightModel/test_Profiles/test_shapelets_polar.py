import lenstronomy.Util.util as util
import numpy as np
import numpy.testing as npt
import pytest
import unittest
from lenstronomy.LightModel.Profiles.shapelets_polar import ShapeletsPolar, ShapeletSetPolar, ShapeletsPolarExp


class TestShapeletsPolar(object):

    def setup(self):
        self.shapelets = ShapeletsPolar()

    def test_function(self):
        x, y = util.make_grid(10, 0.1, 1)
        amp = 1.
        beta = 1.
        n = 5
        m = 0
        complex_bool = False
        flux = self.shapelets.function(x, y, amp, beta, n, m, complex_bool, center_x=0, center_y=0)
        npt.assert_almost_equal(np.sum(flux), 36.3296290765419, decimal=6)

        complex_bool = True
        flux = self.shapelets.function(x, y, amp, beta, n, m, complex_bool, center_x=0, center_y=0)
        npt.assert_almost_equal(np.sum(flux), 0, decimal=6)

        n = 5
        m = 3
        complex_bool = False
        flux = self.shapelets.function(x, y, amp, beta, n, m, complex_bool, center_x=0, center_y=0)
        npt.assert_almost_equal(np.sum(flux), 0, decimal=6)

        complex_bool = True
        flux = self.shapelets.function(x, y, amp, beta, n, m, complex_bool, center_x=0, center_y=0)
        npt.assert_almost_equal(np.sum(flux), 0, decimal=6)

    def test_index2_poly(self):

        index = 0
        n, m, complex_bool = self.shapelets.index2poly(index)
        assert n == 0
        assert m == 0
        assert complex_bool == False

        index = 1
        n, m, complex_bool = self.shapelets.index2poly(index)
        assert n == 1
        assert m == 1
        assert complex_bool == False

        index = 2
        n, m, complex_bool = self.shapelets.index2poly(index)
        assert n == 1
        assert m == 1
        assert complex_bool == True

        index = 3
        n, m, complex_bool = self.shapelets.index2poly(index)
        assert n == 2
        assert m == 0
        assert complex_bool == False

        index = 4
        n, m, complex_bool = self.shapelets.index2poly(index)
        assert n == 2
        assert m == 2
        assert complex_bool == False

        index_new = self.shapelets.poly2index(n, m, complex_bool)
        assert index == index_new

        for index in range(0, 20):
            n, m, complex_bool = self.shapelets.index2poly(index)
            index_new = self.shapelets.poly2index(n, m, complex_bool)
            assert index == index_new

    def test__index2n(self):
        index = 0
        n = self.shapelets._index2n(index)
        assert n == 0
        index = 1
        n = self.shapelets._index2n(index)
        assert n == 1
        index = 2
        n = self.shapelets._index2n(index)
        assert n == 1
        index = 3
        n = self.shapelets._index2n(index)
        assert n == 2


class TestShapeletsPolarExp(object):

    def setup(self):
        self.shapelets = ShapeletsPolarExp()

    def test_function(self):
        x, y = util.make_grid(10, 0.1, 1)
        amp = 1.
        beta = 1.
        n = 2
        m = 0
        complex_bool = False
        flux = self.shapelets.function(x, y, amp, beta, n, m, complex_bool, center_x=0, center_y=0)
        npt.assert_almost_equal(np.sum(flux), 4.704663416542942, decimal=6)

        complex_bool = True
        flux = self.shapelets.function(x, y, amp, beta, n, m, complex_bool, center_x=0, center_y=0)
        npt.assert_almost_equal(np.sum(flux), 0, decimal=6)

        n = 5
        m = 3
        complex_bool = False
        flux = self.shapelets.function(x, y, amp, beta, n, m, complex_bool, center_x=0, center_y=0)
        npt.assert_almost_equal(np.sum(flux), 0, decimal=6)

        complex_bool = True
        flux = self.shapelets.function(x, y, amp, beta, n, m, complex_bool, center_x=0, center_y=0)
        npt.assert_almost_equal(np.sum(flux), 0, decimal=6)

    def test_index2_poly(self):

        index = 0
        n, m, complex_bool = self.shapelets.index2poly(index)
        assert n == 0
        assert m == 0
        assert complex_bool == False

        index = 1
        n, m, complex_bool = self.shapelets.index2poly(index)
        assert n == 1
        assert m == 0
        assert complex_bool == False

        index = 2
        n, m, complex_bool = self.shapelets.index2poly(index)
        assert n == 1
        assert m == 1
        assert complex_bool == True

        index = 3
        n, m, complex_bool = self.shapelets.index2poly(index)
        assert n == 1
        assert m == 1
        assert complex_bool == False

        index = 4
        n, m, complex_bool = self.shapelets.index2poly(index)
        assert n == 2
        assert m == 0
        assert complex_bool == False

        index_new = self.shapelets.poly2index(n, m, complex_bool)
        assert index == index_new

        for index in range(0, 20):
            n, m, complex_bool = self.shapelets.index2poly(index)
            print(n, m, complex_bool, 'test')
            index_new = self.shapelets.poly2index(n, m, complex_bool)
            assert index == index_new

    def test__index2n(self):
        index = 0
        n = self.shapelets._index2n(index)
        assert n == 0
        index = 1
        n = self.shapelets._index2n(index)
        assert n == 1
        index = 2
        n = self.shapelets._index2n(index)
        assert n == 1
        index = 3
        n = self.shapelets._index2n(index)
        assert n == 1


class TestShapeletSetPolar(object):
    """
    class to test Shapelets
    """

    def setup(self):
        self.shapeletSet = ShapeletSetPolar()
        self.shapelets = ShapeletsPolar()
        self.x, self.y = util.make_grid(10, 0.1, 1)

    def test_shapelet_set(self):
        """

        :return:
        """
        n_max = 2
        beta = 1.
        amp = [1, 0, 0, 0, 0, 0]
        output = self.shapeletSet.function(np.array(1), np.array(1), amp, n_max, beta, center_x=0, center_y=0)
        npt.assert_almost_equal(output, 0.20755374871029739, decimal=8)
        input = np.array(0.)
        input += output

        output = self.shapeletSet.function(self.x, self.y, amp, n_max, beta, center_x=0, center_y=0)
        npt.assert_almost_equal(output[10], 0.47957022395315946, decimal=8)
        output = self.shapeletSet.function(1, 1, amp, n_max, beta, center_x=0, center_y=0)
        npt.assert_almost_equal(output, 0.20755374871029739, decimal=8)

        n_max = -1
        beta = 1.
        amp = [1, 0, 0, 0, 0, 0]
        output = self.shapeletSet.function(np.array(1), np.array(1), amp, n_max, beta, center_x=0, center_y=0)
        assert output == 0

    def test_decomposition(self):
        """

        :return:
        """
        n_max = 2
        beta = 10.
        deltaPix = 2
        amp = np.array([1, 1, -1, 1, 1, 1])
        x, y = util.make_grid(100, deltaPix, 1)
        input = self.shapeletSet.function(x, y, amp, n_max, beta, center_x=0, center_y=0)
        amp_out = self.shapeletSet.decomposition(input, x, y, n_max, beta, deltaPix, center_x=0, center_y=0)
        print(amp_out, 'amp_out')
        for i in range(len(amp)):
            print(i, 'i test')
            npt.assert_almost_equal(amp_out[i], amp[i], decimal=4)

    def test_function_split(self):
        n_max = 2
        beta = 10.
        deltaPix = 0.1
        amp = np.array([1, 1, 1, 1, 1, 1])
        x, y = util.make_grid(10, deltaPix, 1)
        function_set = self.shapeletSet.function_split(x, y, amp, n_max, beta, center_x=0, center_y=0)
        test_flux = self.shapelets.function(x, y, amp=1., n=0, m=0, complex_bool=False, beta=beta, center_x=0, center_y=0)
        print(np.shape(function_set))
        print(np.shape(test_flux))
        assert function_set[0][10] == test_flux[10]


class TestShapeletSetPolarExp(object):
    """
    class to test Shapelets
    """

    def setup(self):
        self.shapeletSet = ShapeletSetPolar(exponential=True)
        self.shapelets = ShapeletsPolarExp()
        self.x, self.y = util.make_grid(10, 0.1, 1)

    def test_shapelet_set(self):
        """

        #:return:
        """
        n_max = 2
        beta = 1.
        amp = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        output = self.shapeletSet.function(np.array(1), np.array(1), amp, n_max, beta, center_x=0, center_y=0)
        npt.assert_almost_equal(output, 0.19397908887786985, decimal=8)
        input = np.array(0.)
        input += output

        output = self.shapeletSet.function(self.x, self.y, amp, n_max, beta, center_x=0, center_y=0)
        npt.assert_almost_equal(output[10], 0.4511844400064266, decimal=8)
        output = self.shapeletSet.function(1, 1, amp, n_max, beta, center_x=0, center_y=0)
        npt.assert_almost_equal(output, 0.19397908887786985, decimal=8)

        n_max = -1
        beta = 1.
        amp = [1, 0, 0]
        output = self.shapeletSet.function(np.array(1), np.array(1), amp, n_max, beta, center_x=0, center_y=0)
        assert output == 0

    def test_decomposition(self):
        """

        #:return:
        """
        scale = 10
        n_max = 2
        beta = 1. * scale
        deltaPix = 0.5 * scale
        amp = np.array([1, 1, -1, 1, 1, 1, 1, 1, 1])
        x, y = util.make_grid(1000, deltaPix, 1)
        input = self.shapeletSet.function(x, y, amp, n_max, beta, center_x=0, center_y=0)
        amp_out = self.shapeletSet.decomposition(input, x, y, n_max, beta, deltaPix, center_x=0, center_y=0)
        print(amp_out, 'amp_out')
        for i in range(len(amp)):
            print(self.shapeletSet.shapelets.index2poly(i))
        for i in range(len(amp)):
            print(i, 'i test')
            npt.assert_almost_equal(amp_out[i], amp[i], decimal=2)

    def test_function_split(self):
        n_max = 2
        beta = 10.
        deltaPix = 0.1
        amp = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        x, y = util.make_grid(10, deltaPix, 1)
        function_set = self.shapeletSet.function_split(x, y, amp, n_max, beta, center_x=0, center_y=0)
        test_flux = self.shapelets.function(x, y, amp=1., n=0, m=0, complex_bool=False, beta=beta, center_x=0, center_y=0)
        print(np.shape(function_set))
        print(np.shape(test_flux))
        assert function_set[0][10] == test_flux[10]

    def test_index2poly(self):
        index = 0
        n, m, complex_bool = self.shapeletSet.index2poly(index)
        assert n == 0
        assert m == 0
        assert complex_bool is False


class TestRaise(unittest.TestCase):

    def test_raise(self):
        with self.assertRaises(ValueError):
            shapelets = ShapeletsPolar()
            shapelets.poly2index(n=2, m=1, complex_bool=True)
        with self.assertRaises(ValueError):
            shapelets = ShapeletsPolar()
            shapelets.poly2index(n=2, m=0, complex_bool=True)


if __name__ == '__main__':
    pytest.main()
