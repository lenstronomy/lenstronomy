import pytest
import numpy as np
import numpy.testing as npt
from lenstronomy.Util import sampling_util


def test_unit2gaussian():
    mu, sigma = 5, 1
    cube = np.linspace(0, 1, 3)
    cube = sampling_util.unit2gaussian(cube, mu, sigma)
    npt.assert_equal(cube, [-np.inf, mu, np.inf])

def test_unit2uniform():
    lower, upper = -5, 15
    cube = np.linspace(0, 1, 3)
    cube = sampling_util.unit2uniform(cube, lower, upper)
    npt.assert_equal(cube, [lower, (lower+upper)/2., upper])

def test_uniform2unit():
    lower, upper = -5, 15
    cube = np.linspace(lower, upper, 3)
    cube = sampling_util.uniform2unit(cube, lower, upper)
    npt.assert_equal(cube, [0, 0.5, 1])

def test_cube2args_uniform():
    n_dims = 3
    l, u = -5., 15.
    lowers, uppers = l * np.ones(n_dims), u * np.ones(n_dims)
    truth = [l, (l+u)/2., u]

    cube = [0, 0.5, 1]
    sampling_util.cube2args_uniform(cube, lowers, uppers, n_dims, copy=False)
    npt.assert_equal(cube, truth)

    cube = [0, 0.5, 1]
    sampling_util.cube2args_uniform(cube, lowers, uppers, n_dims, copy=True)
    # they should NOT be equal because cube was not modified in-place
    npt.assert_equal(np.any(np.not_equal(cube, truth)), True)

    cube = sampling_util.cube2args_uniform(cube, lowers, uppers, n_dims, copy=True)
    # here they should
    npt.assert_equal(cube, truth)

def test_cube2args_gaussian():
    n_dims = 3
    l, u = -5., 15.
    m, s = 5., 1.
    lowers, uppers = [l] * n_dims, [u] * n_dims
    means, sigmas  = [m] * n_dims, [s] * n_dims
    truth = [l, m, u]

    cube = [0, 0.5, 1]
    sampling_util.cube2args_gaussian(cube, lowers, uppers, 
                                     means, sigmas, n_dims, copy=False)
    npt.assert_equal(cube, truth)

    cube = [0, 0.5, 1]
    sampling_util.cube2args_gaussian(cube, lowers, uppers, 
                                     means, sigmas, n_dims, copy=True)
    # they should NOT be equal because cube was not modified in-place
    npt.assert_equal(np.any(np.not_equal(cube, truth)), True)

    cube = sampling_util.cube2args_gaussian(cube, lowers, uppers, 
                                            means, sigmas, n_dims, copy=True)
    # here they should
    npt.assert_equal(cube, truth)


if __name__ == '__main__':
    pytest.main()
