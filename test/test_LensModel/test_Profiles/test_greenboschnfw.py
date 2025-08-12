__author__ = "jtekverk"

import numpy as np
import pytest
import numpy.testing as npt
from lenstronomy.LensModel.Profiles.greenboschnfw import GreenBoschNFW
from lenstronomy.LensModel.Profiles.nfw import NFW
import scipy.integrate as si
from lenstronomy.LensModel.lens_model import LensModel


@pytest.mark.parametrize("N", [25, 50])
def test_nfw_limit(N):

    gb = GreenBoschNFW(num_bins=1200, r_min=5e-5, r_max_factor=30.0)
    nfw = NFW()
    Rs, rho0ang, f_b, c_s = 1.0, 300.0, 1.0, 10.0
    alpha_Rs = nfw.rho02alpha(rho0=rho0ang, Rs=Rs)
    xs = np.linspace(0.03, 4.5, N)
    ys = np.linspace(0.03, 4.5, N)
    X, Y = np.meshgrid(xs, ys)
    x, y = X.ravel(), Y.ravel()
    ax_gb, ay_gb = gb.derivatives(x, y, f_b, c_s, Rs, rho0ang, 0.0, 0.0)
    ax_nf, ay_nf = nfw.derivatives(x, y, Rs, alpha_Rs, 0.0, 0.0)
    npt.assert_allclose(ax_gb, ax_nf, rtol=1e-2, atol=1e-6)
    npt.assert_allclose(ay_gb, ay_nf, rtol=1e-2, atol=1e-6)
    f_xx_gb, f_xy_gb, f_yx_gb, f_yy_gb = gb.hessian(x, y, f_b, c_s, Rs, rho0ang, 0, 0)
    f_xx_nf, f_xy_nf, f_yx_nf, f_yy_nf = nfw.hessian(x, y, Rs, alpha_Rs, 0, 0)
    npt.assert_allclose(
        [f_xx_gb, f_xy_gb, f_yx_gb, f_yy_gb],
        [f_xx_nf, f_xy_nf, f_yx_nf, f_yy_nf],
        rtol=3e-2,
        atol=2e-6,
    )


def test_zero_mass_all_zero():

    gb = GreenBoschNFW()
    xs = np.linspace(0.05, 5.0, 80)
    ys = np.linspace(-4.0, 4.0, 90)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    x, y = X.ravel(), Y.ravel()
    ax, ay = gb.derivatives(
        x, y, f_b=1.0, c_s=10.0, Rs=1.0, rho0ang=0.0, center_x=0, center_y=0
    )
    f_xx, f_xy, f_yx, f_yy = gb.hessian(x, y, 1.0, 10.0, 1.0, 0.0, 0, 0)
    npt.assert_allclose([ax, ay, f_xx, f_xy, f_yx, f_yy], 0.0, atol=1e-14)


def test_linearity_rho0ang():

    gb = GreenBoschNFW()
    xs = np.linspace(0.05, 5.0, 80)
    ys = np.linspace(-4.0, 4.0, 90)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    x, y = X.ravel(), Y.ravel()
    ax1, ay1 = gb.derivatives(
        x, y, f_b=0.6, c_s=12.0, Rs=1.3, rho0ang=200.0, center_x=0, center_y=0
    )
    gb.set_dynamic()
    ax2, ay2 = gb.derivatives(
        x, y, f_b=0.6, c_s=12.0, Rs=1.3, rho0ang=500.0, center_x=0, center_y=0
    )
    mask_x = np.abs(ax1) > 1e-12
    mask_y = np.abs(ay1) > 1e-12
    npt.assert_allclose(ax2[mask_x] / ax1[mask_x], 2.5, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(ay2[mask_y] / ay1[mask_y], 2.5, rtol=1e-12, atol=1e-12)


def test_center_shift_invariance():

    gb = GreenBoschNFW()
    cx, cy = 1.8, -1.6
    xs = np.linspace(-5.0, 5.0, 80)
    ys = np.linspace(-4.0, 4.0, 90)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    x, y = X.ravel(), Y.ravel()
    ax1, ay1 = gb.derivatives(
        x + cx,
        y + cy,
        f_b=0.7,
        c_s=15.0,
        Rs=1.0,
        rho0ang=300.0,
        center_x=cx,
        center_y=cy,
    )
    gb.set_dynamic()
    ax2, ay2 = gb.derivatives(
        x, y, f_b=0.7, c_s=15.0, Rs=1.0, rho0ang=300.0, center_x=0, center_y=0
    )
    npt.assert_allclose(ax1, ax2, rtol=1e-12, atol=1e-14)
    npt.assert_allclose(ay1, ay2, rtol=1e-12, atol=1e-14)


def test_hessian_symmetry():

    gb = GreenBoschNFW()
    xs = np.linspace(-5.0, 5.0, 80)
    ys = np.linspace(-4.0, 4.0, 90)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    x, y = X.ravel(), Y.ravel()
    f_xx, f_xy, f_yx, f_yy = gb.hessian(x, y, 0.8, 20.0, 1.0, 200.0, 0, 0)
    npt.assert_allclose(f_xy, f_yx, rtol=1e-13, atol=1e-15)


def test_kappa_monotonic_decrease():

    gb = GreenBoschNFW(num_bins=1200)
    kappa_r, rbin = gb.rbin_kappa_r(f_b=1.0e-5, c_s=0.1, Rs=1.0, rho0ang=100.0)
    diff = np.diff(kappa_r)
    assert np.sum(diff > 1e-10) == 0


def test_cache_and_set_dynamic_consistency():

    gb = GreenBoschNFW()
    k1, r1 = gb.rbin_kappa_r(f_b=0.4, c_s=10.0, Rs=1.1, rho0ang=200.0)
    k2, r2 = gb.rbin_kappa_r(f_b=0.4, c_s=10.0, Rs=1.1, rho0ang=200.0)
    npt.assert_allclose(k1, k2)
    npt.assert_allclose(r1, r2)
    gb.set_dynamic()
    k3, r3 = gb.rbin_kappa_r(f_b=0.4, c_s=10.0, Rs=1.1, rho0ang=200.0)
    assert (k3 is not k1) or (r3 is not r1)
    npt.assert_allclose(k1, k3)
    npt.assert_allclose(r1, r3)


def test_cache_skips_quad(monkeypatch):

    calls = {"n": 0}

    def spy_quad(*a, **k):
        calls["n"] += 1
        return si.quad(*a, **k)

    monkeypatch.setattr("lenstronomy.LensModel.Profiles.greenboschnfw.quad", spy_quad)
    gb = GreenBoschNFW()
    args = dict(f_b=0.4, c_s=10.0, Rs=1.1, rho0ang=200.0)
    gb.rbin_kappa_r(**args)
    first = calls["n"]
    gb.rbin_kappa_r(**args)
    assert calls["n"] == first
    gb.rbin_kappa_r(**{**args, "rho0ang": 201.0})
    assert calls["n"] > first


def test_lensmodel_imports_profile():

    lm = LensModel(lens_model_list=["GreenBoschNFW"])
    kwargs = [
        dict(
            f_b=0.5,
            c_s=12.0,
            Rs=0.5,
            rho0ang=10.0,
            center_x=0.0,
            center_y=0.0,
        )
    ]
    xs = np.linspace(-5.0, 5.0, 80)
    ys = np.linspace(-4.0, 4.0, 90)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    x, y = X.ravel(), Y.ravel()
    pot = lm.potential(x, y, kwargs)
    alpha_x, alpha_y = lm.alpha(x, y, kwargs)
    kappa = lm.kappa(x, y, kwargs)
    assert pot.shape == x.shape
    assert alpha_x.shape == x.shape and alpha_y.shape == x.shape
    assert kappa.shape == x.shape
    assert np.all(np.isfinite(pot))
    assert np.all(np.isfinite(alpha_x))
    assert np.all(np.isfinite(alpha_y))
    assert np.all(np.isfinite(kappa))


@pytest.mark.parametrize("point", [(0.05, 0.02), (-0.03, 0.04)])
def test_derivatives_match_finite_difference(point):

    gb = GreenBoschNFW(num_bins=1200)
    params = dict(f_b=0.5, c_s=12.0, Rs=0.5, rho0ang=10.0, center_x=0.0, center_y=0.0)
    x0, y0 = point
    h = 1e-5
    fx_plus = gb.function(x0 + h, y0, **params)
    fx_minus = gb.function(x0 - h, y0, **params)
    fd_dx = (fx_plus - fx_minus) / (2 * h)
    fy_plus = gb.function(x0, y0 + h, **params)
    fy_minus = gb.function(x0, y0 - h, **params)
    fd_dy = (fy_plus - fy_minus) / (2 * h)
    dpsi_dx, dpsi_dy = gb.derivatives(x0, y0, **params)
    npt.assert_allclose(dpsi_dx, fd_dx, rtol=5e-3, atol=1e-5)
    npt.assert_allclose(dpsi_dy, fd_dy, rtol=5e-3, atol=1e-5)
