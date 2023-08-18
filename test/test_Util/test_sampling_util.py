import pytest
import unittest
import numpy as np
import numpy.testing as npt
from lenstronomy.Util import sampling_util


def test_unit2uniform():
    lower, upper = -5, 15
    cube = np.linspace(0, 1, 3)
    cube = sampling_util.unit2uniform(cube, lower, upper)
    npt.assert_equal(cube, [lower, (lower + upper) / 2.0, upper])


def test_uniform2unit():
    lower, upper = -5, 15
    cube = np.linspace(lower, upper, 3)
    cube = sampling_util.uniform2unit(cube, lower, upper)
    npt.assert_equal(cube, [0, 0.5, 1])


def test_cube2args_uniform():
    n_dims = 3
    l, u = -5.0, 15.0
    lowers, uppers = l * np.ones(n_dims), u * np.ones(n_dims)
    truth = [l, (l + u) / 2.0, u]

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
    l, u = -5.0, 15.0
    m, s = 5.0, 1.0
    lowers, uppers = [l] * n_dims, [u] * n_dims
    means, sigmas = [m] * n_dims, [s] * n_dims
    truth = [l, m, u]

    cube = [0, 0.5, 1]
    sampling_util.cube2args_gaussian(
        cube, lowers, uppers, means, sigmas, n_dims, copy=False
    )
    npt.assert_equal(cube, truth)

    cube = [0, 0.5, 1]
    sampling_util.cube2args_gaussian(
        cube, lowers, uppers, means, sigmas, n_dims, copy=True
    )
    # they should NOT be equal because cube was not modified in-place
    npt.assert_equal(np.any(np.not_equal(cube, truth)), True)

    cube = sampling_util.cube2args_gaussian(
        cube, lowers, uppers, means, sigmas, n_dims, copy=True
    )
    # here they should
    npt.assert_equal(cube, truth)


def test_scale_limits():
    lowers_list, uppers_list = [0, -1, 5], [10, 9, 15]
    lowers, uppers = np.array(lowers_list), np.array(uppers_list)
    widths = uppers - lowers
    scale_factor = 0.5
    lowers_s, uppers_s = sampling_util.scale_limits(
        lowers_list, uppers_list, scale_factor
    )
    npt.assert_equal(lowers_s, np.array([2.5, 1.5, 7.5]))
    npt.assert_equal(uppers_s, np.array([7.5, 6.5, 12.5]))
    npt.assert_equal(widths * scale_factor, (uppers_s - lowers_s))


def test_sample_ball():
    p0 = np.ones(10)
    std = np.ones(10)
    sample = sampling_util.sample_ball(p0, std, size=10000, dist="normal")
    mean = np.mean(sample, axis=0)
    npt.assert_almost_equal(mean, p0, decimal=1)
    sigma = np.std(sample, axis=0)
    npt.assert_almost_equal(sigma, std, decimal=1)

    sample = sampling_util.sample_ball(p0, std, size=10000, dist="uniform")
    mean = np.mean(sample, axis=0)
    npt.assert_almost_equal(mean, p0, decimal=1)
    sigma = np.std(sample, axis=0)
    npt.assert_almost_equal(sigma, std * 0.607, decimal=1)


def test_sample_ball_truncated():
    mean = np.array([0, 1])
    sigma = np.array([0.1, 1])
    lower_limit = np.array([0, 0])
    upper_limit = np.array([3, 4])

    samples = sampling_util.sample_ball_truncated(
        mean, sigma, lower_limit, upper_limit, size=1000
    )

    assert len(samples) == 1000

    var0 = samples[:, 0]
    var1 = samples[:, 1]
    assert np.max(var1) <= upper_limit[1]
    assert np.max(var0) <= upper_limit[0]
    assert np.min(var1) >= lower_limit[1]
    assert np.min(var0) >= lower_limit[0]


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            sampling_util.sample_ball(
                p0=np.ones(10), std=np.ones(10), size=1000, dist="BAD"
            )


if __name__ == "__main__":
    pytest.main()
