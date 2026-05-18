__author__ = "rgh (adapted from sibirrer test_epl)"


import numpy as np
import pytest
import warnings
import numpy.testing as npt
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.bpl import BPL


def _assert_finite_complex(z):
    z = np.asarray(z, dtype=np.complex128)
    assert np.all(np.isfinite(z.real))
    assert np.all(np.isfinite(z.imag))


def _finite(z):
    z = np.asarray(z)
    assert np.all(np.isfinite(z))


class TestBPLInternals(object):
    """Cover scalar-input branches and 3D helper functions that are not hit by BPL-vs-
    EPL tests."""

    def setup_method(self):
        from lenstronomy.LensModel.Profiles.bpl import BPL

        self.bpl = BPL()
        self.major = self.bpl.bpl_major_axis

    @staticmethod
    def _kwargs_bpl(b=1.0, a=2.0, a_c=2.0, r_c=0.7, q=0.8, phi_G=1.0):
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        return dict(
            b=b,
            a=a,
            a_c=a_c,
            r_c=r_c,
            e1=e1,
            e2=e2,
            center_x=0.0,
            center_y=0.0,
        )

    def test_scalar_origin_hits_center_shear_fix_and_major_axis_safe_complex(
        self, monkeypatch
    ):
        """Cover BPL.hessian scalar-origin 'center shear = 0' branch without calling
        BPLMajorAxis.hessian at scalar origin (which is currently not safe due to
        .conj())."""

        # Return an intentionally non-zero shear in major-axis frame,
        # so the wrapper must zero it out at exact center.
        def _fake_major_hessian(x, y, b, a, a_c, r_c, q, **kwargs):
            # f__xx, f__xy, f__yx, f__yy
            return 2.0, 0.3, 0.3, 1.0

        monkeypatch.setattr(self.major, "hessian", _fake_major_hessian)

        # Choose a non-trivial orientation so rotation code still runs
        q = 0.7
        phi_G = 0.9
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)

        kwargs = dict(
            b=1.0,
            a=2.2,
            a_c=1.7,
            r_c=0.6,
            e1=e1,
            e2=e2,
            center_x=0.0,
            center_y=0.0,
        )

        f_xx, f_xy, f_yx, f_yy = self.bpl.hessian(0.0, 0.0, **kwargs)

        # At exact center, wrapper enforces gamma1__=gamma2__=0 => f_xy=0 and f_xx=f_yy=kappa
        npt.assert_allclose(f_xy, 0.0, atol=1e-14)
        npt.assert_allclose(f_yx, 0.0, atol=1e-14)
        npt.assert_allclose(f_xx, f_yy, atol=1e-14)

    def test_beta_func_special_cases(self):
        """Beta_func(a) = Beta(1/2, (a-1)/2).

        Choose a where the Beta function has simple closed forms:
          a=2 -> Beta(1/2, 1/2) = pi
          a=3 -> Beta(1/2, 1)   = 2
        """
        npt.assert_allclose(self.bpl.Beta_func(2.0), np.pi, rtol=1e-12, atol=1e-12)
        npt.assert_allclose(self.bpl.Beta_func(3.0), 2.0, rtol=1e-12, atol=1e-12)

    def test_rho_c_from_b_matches_analytic_for_a2(self):
        """
        rho_c_from_b:
          rho_c = (3-a) * b^(a-1) / (2*B(a)*r_c^a)
        For a=2: B(a)=pi => rho_c = b / (2*pi*r_c^2)
        """
        b = 1.3
        a = 2.0
        r_c = 0.7
        rho = self.bpl.rho_c_from_b(b=b, a=a, r_c=r_c)
        rho_expected = b / (2.0 * np.pi * r_c**2)
        npt.assert_allclose(rho, rho_expected, rtol=1e-12, atol=1e-12)

    def test_mass_3d_lens_inner_outer_vectorized_and_continuous(self):
        """Hit the marked '*'-lines in mass_3d_lens:

        - r = np.asarray(...)
        - allocate M = zeros_like
        - inner/outer masks and both branches
        - m0 term in outer branch
        - return M
        """
        b = 1.0
        a = 2.2
        a_c = 1.6
        r_c = 0.7

        r = np.array([0.2 * r_c, 0.9 * r_c, 1.0 * r_c, 1.1 * r_c, 2.5 * r_c])
        M = self.bpl.mass_3d_lens(r=r, b=b, a=a, a_c=a_c, r_c=r_c)

        assert M.shape == r.shape
        assert np.all(np.isfinite(M))
        assert np.all(M > 0.0)
        assert np.all(np.diff(M) > 0.0)

        # continuity at r_c (function is built to be continuous; derivative may jump)
        eps = 1e-6
        M_minus = self.bpl.mass_3d_lens(r=r_c * (1.0 - eps), b=b, a=a, a_c=a_c, r_c=r_c)
        M_plus = self.bpl.mass_3d_lens(r=r_c * (1.0 + eps), b=b, a=a, a_c=a_c, r_c=r_c)
        npt.assert_allclose(M_minus, M_plus, rtol=1e-5, atol=0.0)

        # scalar r should also work (covers dtype/shape corner cases)
        M_scalar = self.bpl.mass_3d_lens(r=0.3 * r_c, b=b, a=a, a_c=a_c, r_c=r_c)
        assert np.isfinite(M_scalar)

    def test_mass_3d_lens_divergent_slopes_raise(self):
        b = 1.0
        r_c = 0.7

        with pytest.raises(ValueError, match="a_c = 3"):
            self.bpl.mass_3d_lens(r=1.0, b=b, a=2.0, a_c=3.0, r_c=r_c)

        with pytest.raises(ValueError, match="a = 3"):
            self.bpl.mass_3d_lens(r=1.0, b=b, a=3.0, a_c=2.0, r_c=r_c)

    def test_major_axis_hessian_scalar_denom2_exact_zero_regularization(self):
        """Hit the scalar denom2==0 branch in BPLMajorAxis.hessian."""
        b = 1.0
        a = 2.2
        a_c = 1.7

        # Pick values that make denom2 exactly 0 in float64:
        # denom2 = q*x^2 - (1+q^2)*r_c^2  (for y=0)
        q = np.float64(0.3)
        x = np.float64(5.0)
        y = np.float64(0.0)
        r_c = np.sqrt(q * x * x / (1.0 + q * q))  # makes denom2 == 0 exactly

        f_xx, f_xy, f_yx, f_yy = self.major.hessian(
            float(x), float(y), b, a, a_c, float(r_c), float(q)
        )
        assert np.isfinite(f_xx)
        assert np.isfinite(f_xy)
        assert np.isfinite(f_yx)
        assert np.isfinite(f_yy)

    def test_exhyp2f1_scalar_z_wrap_and_general_branch(self):
        """
        Cover exhyp2f1: (1) scalar z -> array([z]) and (2) else-branch (c-a-b != 0.5).
        """
        out = self.major.exhyp2f1(a=0.25, b=0.60, c=2.00, z=0.20)  # c-a-b = 1.15 != 0.5
        # should be array-like (size 1) and finite
        _assert_finite_complex(out)
        assert np.size(out) == 1

    def test_s0arr_scalar_resizes_converges_and_unwraps(self):
        """Cover s0arr scalar-input resize (nzc==1), convergence break, and scalar
        unwrap."""
        out = self.major.s0arr(
            alpha=2.2, alphac=1.7, zel2=0.5, c=0.8, target_precision=1e-4
        )
        _assert_finite_complex(out)

    def test_s2arr_scalar_resizes_converges_and_unwraps(self):
        """Cover s2arr scalar-input resize (nzc==1), convergence break, and scalar
        unwrap."""
        out = self.major.s2arr(
            alpha=2.2, alphac=1.7, zel2=0.5, c=0.8, target_precision=1e-4
        )
        _assert_finite_complex(out)

    def test_major_axis_derivatives_scalar_origin_hits_Z_safe_scalar(self):
        fx, fy = self.major.derivatives(0.0, 0.0, b=1.0, a=2.2, a_c=1.7, r_c=0.6, q=0.8)
        _finite([fx, fy])

    def test_major_axis_hessian_scalar_origin_hits_Z_safe_scalar(self):
        f_xx, f_xy, f_yx, f_yy = self.major.hessian(
            0.0, 0.0, b=1.0, a=2.2, a_c=1.7, r_c=0.6, q=0.8
        )
        _finite([f_xx, f_xy, f_yx, f_yy])
        npt.assert_allclose(f_xy, f_yx, atol=1e-12, rtol=0.0)

    def test_hessian_hits_core_branch_with_small_radius(self):
        # choose point inside r_c so S0/S2 correction is exercised
        q = 0.75
        phi_G = 0.6
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)

        kwargs = dict(
            b=1.0,
            a=2.2,
            a_c=1.3,
            r_c=2.0,
            e1=e1,
            e2=e2,
            center_x=0.0,
            center_y=0.0,
            target_precision=1e-12,
            maxiter=50,
        )

        x = np.array([0.05])
        y = np.array([0.05])

        f_xx, f_xy, f_yx, f_yy = self.bpl.hessian(x, y, **kwargs)
        assert np.all(np.isfinite(f_xx))
        assert np.all(np.isfinite(f_xy))
        assert np.all(np.isfinite(f_yx))
        assert np.all(np.isfinite(f_yy))

    def test_s0arr_s2arr_hit_maxiter_break(self):
        alpha = 2.1
        alphac = 0.6

        # small array, stable values (avoid singularities)
        zel2 = np.array([0.9, 0.8])
        c = np.array([0.3 + 0.0j, 0.4 + 0.0j])

        # eps extremely small + tiny maxiter => guaranteed to stop via maxiter
        out0 = self.major.s0arr(
            alpha, alphac, zel2, c, target_precision=1e-40, maxiter=1
        )
        out2 = self.major.s2arr(
            alpha, alphac, zel2, c, target_precision=1e-40, maxiter=1
        )

        assert np.all(np.isfinite(out0))
        assert np.all(np.isfinite(out2))

    def test_F_hits_a_half_branch(self):
        # Cover F(a,z) special branch a==0.5 (spence-based)
        z = 0.3  # keep within (0,1) to avoid branch cut issues
        val = self.major.F(0.5, z)
        assert np.isfinite(val)

    def test_kappa2func_scalar_returns_float_zero(self):
        # Cover scalar branch: returns python float 0.0 when a==a_c (or r_c<=0)
        out = self.major.kappa2func(b=1.0, a=2.0, a_c=2.0, r_c=1.0, R_el=0.5)
        assert out == 0.0
        assert isinstance(out, float)


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

        Power-law limit is achieved by setting a_c == a (no break in 3D slope), which
        removes the 'core/break' correction terms in the current BPL implementation.
        """
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)

        kwargs_bpl = dict(
            b=b, a=a, a_c=a, r_c=r_c, e1=e1, e2=e2, center_x=0.0, center_y=0.0
        )
        kwargs_epl = dict(theta_E=b, gamma=a, e1=e1, e2=e2, center_x=0.0, center_y=0.0)
        return kwargs_bpl, kwargs_epl

    def test_elliptical_radius_convention(self):
        """Sanity check the two common elliptical-radius definitions mentioned in the
        discussion.

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
            kwargs_bpl, kwargs_epl = self._kwargs_pair(
                b=1.3, a=2.0, q=q, phi_G=1.0, r_c=0.7
            )

            # Ensure **kwargs_bpl passes as a dictionary and not as tuple
            psi_bpl = self.bpl.function(x, y, **kwargs_bpl)
            psi_epl = self.epl.function(x, y, **kwargs_epl)

            dpsi_bpl = psi_bpl[0] - psi_bpl[1]
            dpsi_epl = psi_epl[0] - psi_epl[1]

            npt.assert_almost_equal(dpsi_bpl, dpsi_epl, decimal=4)

    def test_derivatives(self):
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 0.0])

        for q in [1.0, 0.7, 0.4]:
            kwargs_bpl, kwargs_epl = self._kwargs_pair(
                b=1.0, a=2.1, q=q, phi_G=0.3, r_c=0.5
            )

            # Ensure **kwargs_bpl passes as a dictionary and not as tuple
            fx_bpl, fy_bpl = self.bpl.derivatives(x, y, **kwargs_bpl)
            fx_epl, fy_epl = self.epl.derivatives(x, y, **kwargs_epl)

            npt.assert_almost_equal(fx_bpl, fx_epl, decimal=4)
            npt.assert_almost_equal(fy_bpl, fy_epl, decimal=4)

    def test_hessian(self):
        x = np.array([1.0])
        y = np.array([2.0])

        for q in [0.9, 0.7, 0.5]:
            kwargs_bpl, kwargs_epl = self._kwargs_pair(
                b=1.0, a=1.9, q=q, phi_G=1.0, r_c=0.7
            )

            # Ensure **kwargs_bpl passes as a dictionary and not as tuple
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
