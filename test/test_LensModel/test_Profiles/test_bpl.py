__author__ = "rgh (adapted from sibirrer test_epl)"

import numpy as np
import pytest
import warnings
import numpy.testing as npt
import lenstronomy.Util.param_util as param_util


class TestBPLvsEPL(object):
    """Compare BPL against EPL in the pure power-law limit.

    IMPORTANT ON CONVENTIONS
    ------------------------
    - BPLMajorAxis uses elliptical radius R = sqrt(q*x^2 + y^2/q).
    - EPLMajorAxis internally uses R = sqrt(q^2*x^2 + y^2),
      but the public EPL() interface applies a parameter conversion that makes the
      overall model equivalent to the R = sqrt(q*x^2 + y^2/q) convention.

    Therefore, *compare BPL() with EPL()* (not the MajorAxis classes) to avoid
    convention-mismatch issues. This test does exactly that.
    """

    def setup_method(self):
        from lenstronomy.LensModel.Profiles.bpl import BPL
        from lenstronomy.LensModel.Profiles.epl import EPL

        self.bpl = BPL()
        self.epl = EPL()

    @staticmethod
    def _kwargs_pair(b=1.0, a=2.0, q=0.8, phi_G=1.0, r_c=0.7):
        """Return kwargs for BPL and EPL that should match in the power-law limit.

        Power-law limit is achieved by setting a_c == a (no break in 3D slope),
        which removes the 'core/break' correction terms in the current BPL implementation.
        """
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)

        kwargs_bpl = dict(b=b, a=a, a_c=a, r_c=r_c, e1=e1, e2=e2, center_x=0.0, center_y=0.0)
        kwargs_epl = dict(theta_E=b, gamma=a, e1=e1, e2=e2, center_x=0.0, center_y=0.0)
        return kwargs_bpl, kwargs_epl

    def test_elliptical_radius_convention(self):
        """Sanity check the two common elliptical-radius definitions mentioned in the discussion.

        r1 = sqrt(q*x^2 + y^2/q)
        r2 = sqrt(x^2 + y^2/q^2)

        They are related by: r1 = sqrt(q) * r2
        """
        rng = np.random.default_rng(42)
        x = rng.normal(size=10)
        y = rng.normal(size=10)
        q = 0.63

        r1 = np.sqrt(q * x * x + y * y / q)
        r2 = np.sqrt(x * x + y * y / (q * q))

        npt.assert_allclose(r1, np.sqrt(q) * r2, rtol=0, atol=1e-12)

    def test_function(self):
        # Use potential differences to remove the arbitrary additive constant in psi.
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 0.0])

        for q in [0.999, 0.8, 0.4]:
            kwargs_bpl, kwargs_epl = self._kwargs_pair(b=1.3, a=2.0, q=q, phi_G=1.0, r_c=0.7)

            psi_bpl = self.bpl.function(x, y, **kwargs_bpl)
            psi_epl = self.epl.function(x, y, **kwargs_epl)

            dpsi_bpl = psi_bpl[0] - psi_bpl[1]
            dpsi_epl = psi_epl[0] - psi_epl[1]

            npt.assert_almost_equal(dpsi_bpl, dpsi_epl, decimal=4)

    def test_derivatives(self):
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 0.0])

        for q in [1.0, 0.7, 0.4]:
            kwargs_bpl, kwargs_epl = self._kwargs_pair(b=1.0, a=2.1, q=q, phi_G=0.3, r_c=0.5)

            fx_bpl, fy_bpl = self.bpl.derivatives(x, y, **kwargs_bpl)
            fx_epl, fy_epl = self.epl.derivatives(x, y, **kwargs_epl)

            npt.assert_almost_equal(fx_bpl, fx_epl, decimal=4)
            npt.assert_almost_equal(fy_bpl, fy_epl, decimal=4)

    def test_hessian(self):
        x = np.array([1.0])
        y = np.array([2.0])

        for q in [0.9, 0.7, 0.5]:
            kwargs_bpl, kwargs_epl = self._kwargs_pair(b=1.0, a=1.9, q=q, phi_G=1.0, r_c=0.7)

            f_xx, f_xy, f_yx, f_yy = self.bpl.hessian(x, y, **kwargs_bpl)
            f_xx_e, f_xy_e, f_yx_e, f_yy_e = self.epl.hessian(x, y, **kwargs_epl)

            npt.assert_almost_equal(f_xx, f_xx_e, decimal=4)
            npt.assert_almost_equal(f_yy, f_yy_e, decimal=4)
            npt.assert_almost_equal(f_xy, f_xy_e, decimal=4)
            npt.assert_almost_equal(f_xy, f_yx, decimal=8)

    def test_static(self):
        # Use numpy arrays to follow the vectorized code path (BPL uses .conj on numpy complex).
        x = np.array([1.0])
        y = np.array([1.0])
        phi_G, q = 0.3, 0.8
        kwargs_bpl, _ = self._kwargs_pair(b=1.1, a=2.0, q=q, phi_G=phi_G, r_c=0.7)

        f_ = self.bpl.function(x, y, **kwargs_bpl)
        self.bpl.set_static(**kwargs_bpl)
        f_static = self.bpl.function(x, y, **kwargs_bpl)

        npt.assert_almost_equal(f_, f_static, decimal=8)

        self.bpl.set_dynamic()
        kwargs_bpl2, _ = self._kwargs_pair(b=2.0, a=2.2, q=q, phi_G=phi_G, r_c=0.7)
        f_dyn = self.bpl.function(x, y, **kwargs_bpl2)
        assert not np.allclose(f_dyn, f_static)

    def test_regularization(self):
        # origin should be well-defined (no NaNs/inf)
        b, a = 1.0, 2.0
        q, phi_G = 0.8, 1.0
        kwargs_bpl, _ = self._kwargs_pair(b=b, a=a, q=q, phi_G=phi_G, r_c=0.7)

        x0 = np.array([0.0])
        y0 = np.array([0.0])

        # For singular slopes (e.g. a~2), the deflection at r=0 is not uniquely defined.
        # Implementations regularize divisions by substituting a tiny complex number; that can
        # trigger benign RuntimeWarnings (e.g. inf-inf) in intermediate special functions.
        # Here we only assert finiteness and symmetry properties at the exact origin.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            f_x0, f_y0 = self.bpl.derivatives(x0, y0, **kwargs_bpl)
            assert np.all(np.isfinite(f_x0))
            assert np.all(np.isfinite(f_y0))

            f_xx0, f_xy0, f_yx0, f_yy0 = self.bpl.hessian(x0, y0, **kwargs_bpl)
            assert np.all(np.isfinite(f_xx0))
            assert np.all(np.isfinite(f_yy0))
            assert np.all(np.isfinite(f_xy0))
            assert np.all(np.isfinite(f_yx0))

        npt.assert_almost_equal(f_xy0, f_yx0, decimal=10)
        npt.assert_almost_equal(f_xy0, 0.0, decimal=6)

        # Avoid exactly r=0 and also avoid the extreme z->1 regime inside kappa2func that can
        # produce inf-inf when a==a_c. We probe a small but not *too* small radius instead.
        r_c = kwargs_bpl["r_c"]
        eps = 1e-2 * r_c
        x = np.array([eps])
        y = np.array([0.0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            f_xx, f_xy, f_yx, f_yy = self.bpl.hessian(x, y, **kwargs_bpl)

        kappa = 0.5 * (f_xx + f_yy)
        assert np.all(np.isfinite(kappa))
        # In the singular isothermal-like case (a≈2), kappa should be non-zero at finite radius.
        assert np.all(np.abs(kappa) > 10.0)

        npt.assert_almost_equal(f_xy, f_yx, decimal=10)
        npt.assert_almost_equal(f_xy, 0.0, decimal=6)


if __name__ == "__main__":
    pytest.main()
