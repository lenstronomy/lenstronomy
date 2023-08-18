import numpy as np
import pytest
import numpy.testing as npt
from lenstronomy.Util import util
import lenstronomy.Util.param_util as param_util


def test_cart2polar():
    # singel 2d coordinate transformation
    center_x, center_y = 0, 0
    x = 1
    y = 1
    r, phi = param_util.cart2polar(x, y, center_x, center_y)
    assert r == np.sqrt(2)  # radial part
    assert phi == np.arctan(1)
    # array of 2d coordinates
    x = np.array([1, 2])
    y = np.array([1, 1])

    r, phi = param_util.cart2polar(x, y, center_x, center_y)
    assert r[0] == np.sqrt(2)  # radial part
    assert phi[0] == np.arctan(1)


def test_polar2cart():
    # singel 2d coordinate transformation
    center = np.array([0, 0])
    r = 1
    phi = np.pi
    x, y = param_util.polar2cart(r, phi, center)
    assert x == -1
    assert abs(y) < 10e-14


def test_phi_q2_ellipticity():
    phi, q = 0, 1
    e1, e2 = param_util.phi_q2_ellipticity(phi, q)
    assert e1 == 0
    assert e2 == 0

    phi, q = 1, 1
    e1, e2 = param_util.phi_q2_ellipticity(phi, q)
    assert e1 == 0
    assert e2 == 0

    phi, q = 2.0, 0.95
    e1, e2 = param_util.phi_q2_ellipticity(phi, q)
    npt.assert_almost_equal(e1, -0.016760092842656733, decimal=8)
    npt.assert_almost_equal(e2, -0.019405192187382792, decimal=8)

    phi, q = 0, 0.9
    e1, e2 = param_util.phi_q2_ellipticity(phi, q)
    npt.assert_almost_equal(e1, 0.05263157894736841, decimal=8)
    assert e2 == 0


def test_ellipticity2phi_q():
    e1, e2 = 0.3, 0
    phi, q = param_util.ellipticity2phi_q(e1, e2)
    assert phi == 0
    npt.assert_almost_equal(q, 0.53846153846153844, decimal=8)

    # Works on np arrays as well
    e1 = np.array([0.3, 0.9])
    e2 = np.array([0.0, 0.9])
    phi, q = param_util.ellipticity2phi_q(e1, e2)
    assert np.allclose(phi, [0.0, 0.39269908], atol=1.0e-08)
    assert np.allclose(q, [0.53846153, 5.00025001e-05], atol=1.0e-08)


def test_ellipticity2phi_q_symmetry():
    phi, q = 1.5, 0.8
    e1, e2 = param_util.phi_q2_ellipticity(phi, q)
    phi_new, q_new = param_util.ellipticity2phi_q(e1, e2)
    npt.assert_almost_equal(phi, phi_new, decimal=8)
    npt.assert_almost_equal(q, q_new, decimal=8)

    phi, q = -1.5, 0.8
    e1, e2 = param_util.phi_q2_ellipticity(phi, q)
    phi_new, q_new = param_util.ellipticity2phi_q(e1, e2)
    npt.assert_almost_equal(phi, phi_new, decimal=8)
    npt.assert_almost_equal(q, q_new, decimal=8)

    e1, e2 = 0.1, -0.1
    phi, q = param_util.ellipticity2phi_q(e1, e2)
    e1_new, e2_new = param_util.phi_q2_ellipticity(phi, q)
    npt.assert_almost_equal(e1, e1_new, decimal=10)
    npt.assert_almost_equal(e2, e2_new, decimal=10)

    e1, e2 = 2.99, -0.0
    phi, q = param_util.ellipticity2phi_q(e1, e2)
    print(phi, q)
    e1_new, e2_new = param_util.phi_q2_ellipticity(phi, q)
    phi_new, q_new = param_util.ellipticity2phi_q(e1_new, e2_new)
    npt.assert_almost_equal(phi, phi_new, decimal=10)
    npt.assert_almost_equal(q, q_new, decimal=10)
    # npt.assert_almost_equal(e1, e1_new, decimal=10)
    # npt.assert_almost_equal(e2, e2_new, decimal=10)


def test_transform_e1e2():
    e1 = 0.01
    e2 = 0.0
    x = 0.0
    y = 1.0
    x_, y_ = param_util.transform_e1e2_product_average(
        x, y, e1, e2, center_x=0, center_y=0
    )
    x_new = (1 - e1) * x - e2 * y
    y_new = -e2 * x + (1 + e1) * y
    det = np.sqrt((1 - e1) * (1 + e1) + e2**2)
    npt.assert_almost_equal(x_, x_new / det, decimal=5)
    npt.assert_almost_equal(y_, y_new / det, decimal=5)


def test_phi_gamma_ellipticity():
    phi = -1.0
    gamma = 0.1
    e1, e2 = param_util.shear_polar2cartesian(phi, gamma)
    print(e1, e2, "e1, e2")
    phi_out, gamma_out = param_util.shear_cartesian2polar(e1, e2)
    npt.assert_almost_equal(phi_out, phi, decimal=8)
    npt.assert_almost_equal(gamma_out, gamma_out, decimal=8)


def test_phi_gamma_ellipticity_2():
    e1, e2 = -0.04, -0.01
    phi, gamma = param_util.shear_cartesian2polar(e1, e2)

    e1_out, e2_out = param_util.shear_polar2cartesian(phi, gamma)
    npt.assert_almost_equal(e1, e1_out, decimal=10)
    npt.assert_almost_equal(e2, e2_out, decimal=10)


def test_displace_eccentricity():
    x, y = util.make_grid(numPix=10, deltapix=1)
    e1 = 0.1
    e2 = -0
    center_x, center_y = 0, 0
    x_, y_ = param_util.transform_e1e2_product_average(
        x, y, e1, e2, center_x=center_x, center_y=center_y
    )

    phi_G, q = param_util.ellipticity2phi_q(e1, e2)
    x_shift = x - center_x
    y_shift = y - center_y

    cos_phi = np.cos(phi_G)
    sin_phi = np.sin(phi_G)
    print(cos_phi, sin_phi)

    xt1 = cos_phi * x_shift + sin_phi * y_shift
    xt2 = -sin_phi * x_shift + cos_phi * y_shift
    xt1 *= np.sqrt(q)
    xt2 /= np.sqrt(q)
    npt.assert_almost_equal(x_, xt1, decimal=8)
    npt.assert_almost_equal(y_, xt2, decimal=8)

    x, y = np.array([1, 0]), np.array([0, 1])
    e1 = 0.1
    e2 = 0
    center_x, center_y = 0, 0
    x_, y_ = param_util.transform_e1e2_product_average(
        x, y, e1, e2, center_x=center_x, center_y=center_y
    )

    phi_G, q = param_util.ellipticity2phi_q(e1, e2)
    x_shift = x - center_x
    y_shift = y - center_y

    cos_phi = np.cos(phi_G)
    sin_phi = np.sin(phi_G)
    print(cos_phi, sin_phi)

    xt1 = cos_phi * x_shift + sin_phi * y_shift
    xt2 = -sin_phi * x_shift + cos_phi * y_shift
    xt1 *= np.sqrt(q)
    xt2 /= np.sqrt(q)
    npt.assert_almost_equal(x_, xt1, decimal=8)
    npt.assert_almost_equal(y_, xt2, decimal=8)


def test_transform_e1e2_square_average():
    x, y = np.array([1, 0]), np.array([0, 1])
    e1 = 0.1
    e2 = 0
    center_x, center_y = 0, 0

    x_, y_ = param_util.transform_e1e2_square_average(
        x, y, e1, e2, center_x=center_x, center_y=center_y
    )
    npt.assert_almost_equal(
        np.sum(x**2 + y**2), np.sum(x_**2 + y_**2), decimal=8
    )


if __name__ == "__main__":
    pytest.main()
