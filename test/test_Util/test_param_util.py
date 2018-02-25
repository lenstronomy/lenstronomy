import numpy as np
import pytest
import numpy.testing as npt
import lenstronomy.Util.param_util as param_util


def test_cart2polar():
    #singel 2d coordinate transformation
    center = np.array([0,0])
    x = 1
    y = 1
    r, phi = param_util.cart2polar(x,y,center)
    assert r == np.sqrt(2) #radial part
    assert phi == np.arctan(1)
    #array of 2d coordinates
    center = np.array([0,0])
    x = np.array([1,2])
    y = np.array([1,1])

    r, phi = param_util.cart2polar(x,y,center)
    assert r[0] == np.sqrt(2) #radial part
    assert phi[0] == np.arctan(1)


def test_polar2cart():
    #singel 2d coordinate transformation
    center = np.array([0,0])
    r = 1
    phi = np.pi
    x, y = param_util.polar2cart(r, phi, center)
    assert x == -1
    assert abs(y) < 10e-14


def test_phi_q2_elliptisity():
    phi, q = 0, 1
    e1,e2 = param_util.phi_q2_ellipticity(phi, q)
    assert e1 == 0
    assert e2 == 0

    phi, q = 1, 1
    e1,e2 = param_util.phi_q2_ellipticity(phi, q)
    assert e1 == 0
    assert e2 == 0

    phi, q = 2.,0.95
    e1, e2 = param_util.phi_q2_ellipticity(phi, q)
    assert e1 == -0.016760092842656733
    assert e2 == -0.019405192187382792


def test_phi_q2_elliptisity_bounds():
    bounds = 'lower'
    phi, q = 0, 1
    e1, e2 = param_util.phi_q2_elliptisity_bounds(phi, q, bounds)
    assert e1 == 0
    assert e2 == 0

    phi, q = 1, 1
    e1, e2 = param_util.phi_q2_elliptisity_bounds(phi,q, bounds)
    assert e1 == 0
    assert e2 == 0

    phi, q = 2., 0.95
    e1, e2 = param_util.phi_q2_elliptisity_bounds(phi, q, bounds)
    assert e1 == -0.019405192187382792
    assert e2 == -0.019405192187382792

    bounds = 'upper'
    phi, q = 2., 0.95
    e1, e2 = param_util.phi_q2_elliptisity_bounds(phi, q, bounds)
    assert e1 == 0.019405192187382792
    assert e2 == 0.019405192187382792


def test_elliptisity2phi_q():
    e1, e2 = 0.3,0
    phi,q = param_util.elliptisity2phi_q(e1,e2)
    assert phi == 0
    assert q == 0.53846153846153844


def test_elliptisity2phi_q_symmetry():
    phi,q = 1.5, 0.8
    e1,e2 = param_util.phi_q2_ellipticity(phi, q)
    phi_new,q_new = param_util.elliptisity2phi_q(e1,e2)
    assert phi == phi_new
    assert q == q_new

    phi,q = -1.5, 0.8
    e1,e2 = param_util.phi_q2_ellipticity(phi, q)
    phi_new,q_new = param_util.elliptisity2phi_q(e1,e2)
    assert phi == phi_new
    assert q == q_new

    e1, e2 = 0.1, -0.1
    phi, q = param_util.elliptisity2phi_q(e1, e2)
    e1_new, e2_new = param_util.phi_q2_ellipticity(phi, q)
    npt.assert_almost_equal(e1, e1_new, decimal=10)
    npt.assert_almost_equal(e2, e2_new, decimal=10)

    e1, e2 = 2.99, -0.0
    phi, q = param_util.elliptisity2phi_q(e1, e2)
    print(phi, q)
    e1_new, e2_new = param_util.phi_q2_ellipticity(phi, q)
    phi_new, q_new = param_util.elliptisity2phi_q(e1_new, e2_new)
    npt.assert_almost_equal(phi, phi_new, decimal=10)
    npt.assert_almost_equal(q, q_new, decimal=10)
    #npt.assert_almost_equal(e1, e1_new, decimal=10)
    #npt.assert_almost_equal(e2, e2_new, decimal=10)



def test_phi_gamma_ellipticity():
    phi = -1.
    gamma = 0.1
    e1, e2 = param_util.phi_gamma_ellipticity(phi, gamma)
    print(e1, e2, 'e1, e2')
    phi_out, gamma_out = param_util.ellipticity2phi_gamma(e1, e2)
    assert phi == phi_out
    assert gamma == gamma_out


if __name__ == '__main__':
    pytest.main()