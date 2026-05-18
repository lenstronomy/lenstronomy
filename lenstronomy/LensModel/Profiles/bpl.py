__author__ = "Guanhua Rui, Wei Du"

import warnings

import numpy as np
from scipy.integrate import quad
from scipy.special import beta, hyp2f1, spence

import lenstronomy.Util.param_util as param_util
import lenstronomy.Util.util as util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["BPL", "BPLMajorAxis"]

# tiny number used to avoid divisions by zero at the exact origin / on branch cuts
_TINY = 1e-15

# suppress warnings from intermediate complex algebra (we sanitize outputs with nan_to_num)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class BPL(LensProfileBase):
    """Broken Power Law (BPL) mass profile.

    The 3D density follows Du et al. (2020):

    .. math::
        \\rho(r) = \\begin{cases}
        \\rho_c\\,(r/r_c)^{-\\alpha_c} & r \\le r_c \\\\
        \\rho_c\\,(r/r_c)^{-\\alpha} & r \\ge r_c
        \\end{cases}

    Assuming a homoeoidal (elliptically symmetric) projected mass distribution, the convergence can be written as:

    .. math::
        \\kappa(R) = \\kappa_1(R) + \\kappa_2(R),

    with a elliptical power-law term

    .. math::
        \\kappa_1(R) = \\frac{3-\\alpha}{2}\\left(\\frac{b}{R}\\right)^{\\alpha-1},

    and a core/break correction \\(\\kappa_2\\) that is non-zero only for \\(R\\le r_c\\) (Du et al. 2020).

    The elliptical radius is defined in the lenstronomy convention as

    .. math::
        R \\equiv R_{\\rm el} = \\sqrt{q x^2 + y^2/q},

    where \\(q\\) is the minor/major axis ratio and \\((x, y)\\) are coordinates in the major-axis-aligned frame.

    The analytic deflection and shear expressions are implemented in :class:`~lenstronomy.LensModel.Profiles.bpl.BPLMajorAxis`
    using the complex BK75 formalism and the special series \\(S_0\\) and \\(S_2\\) defined in Du et al. (2020).
    The lensing potential is evaluated via 1D numerical quadrature.
    """

    param_names = ["b", "a", "a_c", "r_c", "e1", "e2", "center_x", "center_y"]
    lower_limit_default = {
        "b": 0,
        "a": 1,
        "a_c": 0,
        "r_c": 0,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "b": 100,
        "a": 3,
        "a_c": 3,
        "r_c": 100,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(self):
        """Initialize the BPL profile."""

        self.bpl_major_axis = BPLMajorAxis()
        super(BPL, self).__init__()

    # -------- parameter conversion --------

    def param_conv(self, b, a, a_c, r_c, e1, e2):
        """Converts parameters as defined in this class to the parameters used in the
        BPLMajorAxis() class.

        :param b: lens strength parameter as defined in the profile class
        :param a: outer slope \\(\\alpha\\)
        :param a_c: inner slope \\(\\alpha_c\\)
        :param r_c: break radius \\(r_c\\)
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :return: b, a, a_c, r_c, q, phi_G
        """

        if self._static is True:
            return (
                self._b_static,
                self._a_static,
                self._a_c_static,
                self._r_c_static,
                self._q_static,
                self._phi_G_static,
            )
        return self._param_conv(b, a, a_c, r_c, e1, e2)

    @staticmethod
    def _param_conv(b, a, a_c, r_c, e1, e2):
        """Convert (e1, e2) ellipticity to (q, phi_G) and return arguments for
        BPLMajorAxis.

        :param b: lens strength parameter
        :param a: outer slope \\(\\alpha\\)
        :param a_c: inner slope \\(\\alpha_c\\)
        :param r_c: break radius \\(r_c\\)
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :return: b, a, a_c, r_c, q, phi_G
        """

        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        return b, a, a_c, r_c, q, phi_G

    def set_static(self, b, a, a_c, r_c, e1, e2, center_x=0, center_y=0):
        """Cache converted parameters for repeated calls.

        :param b: lens strength parameter
        :param a: outer 3D slope \\(\\alpha\\)
        :param a_c: inner 3D slope \\(\\alpha_c\\)
        :param r_c: break radius \\(r_c\\)
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :param center_x: profile center
        :param center_y: profile center
        :return: self variables set
        """

        self._static = True
        (
            self._b_static,
            self._a_static,
            self._a_c_static,
            self._r_c_static,
            self._q_static,
            self._phi_G_static,
        ) = self._param_conv(b, a, a_c, r_c, e1, e2)

    def set_dynamic(self):
        """Disable static-parameter caching.

        :return: None
        """

        self._static = False
        for name in [
            "_b_static",
            "_a_static",
            "_a_c_static",
            "_r_c_static",
            "_phi_G_static",
            "_q_static",
        ]:
            if hasattr(self, name):
                delattr(self, name)

    # -------- lensing quantities --------

    def function(
        self,
        x,
        y,
        b,
        a,
        a_c,
        r_c,
        e1,
        e2,
        center_x=0,
        center_y=0,
        target_precision=None,
        maxiter=None,
        **kwargs,
    ):
        """Returns the lensing potential.

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param b: lens strength parameter
        :param a: outer slope \\(\\alpha\\)
        :param a_c: inner slope \\(\\alpha_c\\)
        :param r_c: break radius \\(r_c\\)
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :param target_precision: convergence threshold for internal S0/S2 series
            evaluation
        :param maxiter: maximum iteration cap for internal S0/S2 series evaluation
        :return: lensing potential
        """

        b, a, a_c, r_c, q, phi_G = self.param_conv(b, a, a_c, r_c, e1, e2)
        x_ = x - center_x
        y_ = y - center_y
        x__, y__ = util.rotate(x_, y_, phi_G)
        return self.bpl_major_axis.function(
            x__,
            y__,
            b,
            a,
            a_c,
            r_c,
            q,
            target_precision=target_precision,
            maxiter=maxiter,
        )

    def derivatives(
        self,
        x,
        y,
        b,
        a,
        a_c,
        r_c,
        e1,
        e2,
        center_x=0,
        center_y=0,
        target_precision=None,
        maxiter=None,
        **kwargs,
    ):
        """Returns the deflection angles.

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param b: lens strength parameter
        :param a: outer slope \\(\\alpha\\)
        :param a_c: inner slope \\(\\alpha_c\\)
        :param r_c: break radius \\(r_c\\)
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :param target_precision: convergence threshold for internal S0 series evaluation
        :param maxiter: maximum iteration cap for internal S0 series evaluation
        :return: alpha_x, alpha_y
        """

        b, a, a_c, r_c, q, phi_G = self.param_conv(b, a, a_c, r_c, e1, e2)
        x_ = x - center_x
        y_ = y - center_y
        x__, y__ = util.rotate(x_, y_, phi_G)

        f__x, f__y = self.bpl_major_axis.derivatives(
            x__,
            y__,
            b,
            a,
            a_c,
            r_c,
            q,
            target_precision=target_precision,
            maxiter=maxiter,
        )
        f_x, f_y = util.rotate(f__x, f__y, -phi_G)
        return f_x, f_y

    def hessian(
        self,
        x,
        y,
        b,
        a,
        a_c,
        r_c,
        e1,
        e2,
        center_x=0,
        center_y=0,
        target_precision=None,
        maxiter=None,
        **kwargs,
    ):
        """Hessian matrix of the lensing potential.

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param b: lens strength parameter
        :param a: outer slope \\(\\alpha\\)
        :param a_c: inner slope \\(\\alpha_c\\)
        :param r_c: break radius \\(r_c\\)
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :param target_precision: convergence threshold for internal S2 series evaluation
        :param maxiter: maximum iteration cap for internal S2 series evaluation
        :return: f_xx, f_xy, f_yx, f_yy
        """

        b, a, a_c, r_c, q, phi_G = self.param_conv(b, a, a_c, r_c, e1, e2)
        x_ = x - center_x
        y_ = y - center_y
        x__, y__ = util.rotate(x_, y_, phi_G)

        f__xx, f__xy, f__yx, f__yy = self.bpl_major_axis.hessian(
            x__,
            y__,
            b,
            a,
            a_c,
            r_c,
            q,
            target_precision=target_precision,
            maxiter=maxiter,
        )

        # rotate shear components back to the original frame
        kappa = 0.5 * (f__xx + f__yy)
        gamma1__ = 0.5 * (f__xx - f__yy)
        gamma2__ = f__xy

        # --- force shear to vanish exactly at the center (supports scalar & array) ---
        x_arr = np.asarray(x__)
        y_arr = np.asarray(y__)
        mask0 = (x_arr == 0) & (y_arr == 0)

        if np.ndim(mask0) == 0:
            if bool(mask0):
                gamma1__ = 0.0 * gamma1__
                gamma2__ = 0.0 * gamma2__
        else:
            gamma1__ = np.array(gamma1__, copy=True)
            gamma2__ = np.array(gamma2__, copy=True)
            gamma1__[mask0] = 0.0
            gamma2__[mask0] = 0.0

        gamma1 = np.cos(2 * phi_G) * gamma1__ - np.sin(2 * phi_G) * gamma2__
        gamma2 = np.sin(2 * phi_G) * gamma1__ + np.cos(2 * phi_G) * gamma2__
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy

    # -------- helper utilities for 3D mass (optional) --------

    @staticmethod
    def Beta_func(a):
        """Beta function coefficient \\(B(\\alpha)\\) defined in Du et al. (2020, Eq.
        5).

        :param a: outer slope \\(\\alpha\\)
        :return: \\(B(\\alpha)=\\mathrm{B}(1/2,(\\alpha-1)/2)\\)
        """

        return beta(1 / 2, (a - 1) / 2)

    def rho_c_from_b(self, b, a, r_c):
        """Compute \\(\\rho_c\\) from (b, a, r_c) using Du et al. (2020, Eq. 8), in lens
        units.

        The normalization is defined such that \\(b^{\\alpha-1} = B(\\alpha)\\,2/(3-\\alpha)\\,\\rho_c\\,r_c^{\\alpha}\\).

        :param b: lens strength parameter
        :param a: outer slope \\(\\alpha\\)
        :param r_c: break radius \\(r_c\\)
        :return: central density \\(\\rho_c\\) (in internal lensing units)
        """

        B_a = self.Beta_func(a)
        return (3.0 - a) * b ** (a - 1.0) / (2.0 * B_a * r_c**a)

    def mass_3d_lens(self, r, b, a, a_c, r_c, e1=None, e2=None):
        """3D enclosed mass \\(M(<r)\\) for the BPL density profile.

        :param r: 3D radius (can be scalar or array)
        :param b: lens strength parameter
        :param a: outer slope \\(\\alpha\\)
        :param a_c: inner slope \\(\\alpha_c\\)
        :param r_c: break radius \\(r_c\\)
        :param e1: eccentricity component (not used)
        :param e2: eccentricity component (not used)
        :return: enclosed 3D mass \\(M(<r)\\)
        """

        r = np.asarray(r, dtype=float)

        rho_c = self.rho_c_from_b(b, a, r_c)
        M = np.zeros_like(r, dtype=float)

        if np.isclose(3.0 - a_c, 0.0):
            raise ValueError("a_c = 3 causes divergent inner mass.")
        if np.isclose(3.0 - a, 0.0):
            raise ValueError("a = 3 causes divergent outer mass.")

        inner = r <= r_c
        outer = ~inner

        if np.any(inner):
            r_in = r[inner]
            M[inner] = (
                4.0 * np.pi * rho_c * r_c**a_c * r_in ** (3.0 - a_c) / (3.0 - a_c)
            )

        if np.any(outer):
            r_out = r[outer]
            # Du+2020 Eq. (3)
            m0 = -4.0 * np.pi * rho_c * (a - a_c) * r_c**3 / ((3.0 - a) * (3.0 - a_c))
            M[outer] = (
                4.0 * np.pi * rho_c * r_c**a * r_out ** (3.0 - a) / (3.0 - a) + m0
            )

        return M


class BPLMajorAxis(LensProfileBase):
    """This class contains the function and the derivatives of the elliptical BPL
    profile.

    In the major-axis-aligned frame, the complex conjugate deflection field for an elliptically symmetric surface density
    can be written (Bourassa & Kantowski 1975; Du et al. 2020)

    .. math::
        \\alpha^*(z)=\\frac{2}{z}\\int_0^{R_{\\rm el}}\\frac{\\kappa(R)\\,R\\,dR}{\\sqrt{1-\\zeta^2 R^2}},\\quad
        \\zeta^2=(1/q-q)/z^2,

    where \\(z=x+i y\\) and \\(R_{\\rm el}=\\sqrt{q x^2+y^2/q}\\). Substituting the BPL convergence yields closed-form
    expressions for the deflection and shear (Du et al. 2020, Eq. 18-26). The inner correction terms are evaluated via the
    series \\(S_0\\) (deflection) and \\(S_2\\) (shear), using Aitken-accelerated recursion for numerical stability.
    """

    param_names = ["b", "a", "a_c", "r_c", "center_x", "center_y"]

    DEFAULT_TARGET_PRECISION = 1e-5
    DEFAULT_MAXITER = 4000

    def __init__(
        self, target_precision=DEFAULT_TARGET_PRECISION, maxiter=DEFAULT_MAXITER
    ):
        """Create a major-axis-aligned BPL evaluator.

        :param target_precision: default convergence threshold passed to S0/S2 when not
            overridden per call
        :param maxiter: default maximum iteration count passed to S0/S2 when not
            overridden per call
        :return: self
        """

        self._target_precision = float(target_precision)
        self._maxiter = int(maxiter)
        super(BPLMajorAxis, self).__init__()

    # ----------------- public API -----------------

    def function(
        self,
        x,
        y,
        b,
        a,
        a_c,
        r_c,
        q,
        target_precision=None,
        maxiter=None,
        **kwargs,
    ):
        """Returns the lensing potential (computed via 1D numerical quadrature).

        :param x: x-coordinate in image plane relative to center (major axis frame)
        :param y: y-coordinate in image plane relative to center (major axis frame)
        :param b: lens strength parameter
        :param a: outer slope \\(\\alpha\\)
        :param a_c: inner slope \\(\\alpha_c\\)
        :param r_c: break radius \\(r_c\\)
        :param q: axis ratio (minor/major)
        :param target_precision: convergence threshold for internal series (passed
            through)
        :param maxiter: maximum iteration cap for internal series (passed through)
        :return: lensing potential
        """

        vectorized_quad = np.vectorize(
            lambda aa, bb: quad(
                self.integrand_psi, 0.0, 1.0, args=(aa, bb, a, a_c, b, r_c, q)
            )[0]
        )
        return vectorized_quad(x, y)

    def derivatives(
        self,
        x,
        y,
        b,
        a,
        a_c,
        r_c,
        q,
        target_precision=None,
        maxiter=None,
    ):
        """Returns the deflection angles.

        The deflection is evaluated as \\(\\alpha^*=\\alpha_1^*+\\alpha_2^*\\) following
        Du et al. (2020, Eq. 18-22), where \\(\\alpha_1\\) is the EPL (Elliptical Power
        Law) term and \\(\\alpha_2\\) is the inner correction involving the series
        \\(S_0\\).

        :param x: x-coordinate in image plane relative to center (major axis frame)
        :param y: y-coordinate in image plane relative to center (major axis frame)
        :param b: lens strength parameter
        :param a: outer slope \\(\\alpha\\)
        :param a_c: inner slope \\(\\alpha_c\\)
        :param r_c: break radius \\(r_c\\)
        :param q: axis ratio (minor/major)
        :param target_precision: convergence threshold for S0 recursion
        :param maxiter: maximum iteration cap for S0 recursion
        :return: alpha_x, alpha_y
        """

        tp, mi = self._resolve_settings(target_precision, maxiter)
        geom = self._prepare_geometry(x, y, q)

        pref = self._epl_prefactors(b=b, a=a, R_el=geom["R_el"], invZ=geom["invZ"])
        alpha1 = self._alpha1_epl_like(
            base1=pref["base1"], pow_a=pref["pow_a"], a=a, U_R=geom["U_R"]
        )

        alpha2 = 0.0
        if self._core_active(r_c=r_c, a=a, a_c=a_c):
            core = self._core_precompute(
                invZ=geom["invZ"],
                invZ2=geom["invZ2"],
                Z=geom["Z"],
                z2=geom["z2"],
                q=q,
                b=b,
                a=a,
                a_c=a_c,
                r_c=r_c,
                R_el=geom["R_el"],
                Beta_a=pref["Beta_a"],
                target_precision=tp,
                maxiter=mi,
                need_S0=True,
                need_S2=False,
            )
            alpha2 = core["alpha2"]

        alpha = alpha1 + alpha2
        alpha_x = np.nan_to_num(alpha.real, posinf=1e15, neginf=-1e15)
        alpha_y = np.nan_to_num(-alpha.imag, posinf=1e15, neginf=-1e15)
        return alpha_x, alpha_y

    def hessian(
        self,
        x,
        y,
        b,
        a,
        a_c,
        r_c,
        q,
        target_precision=None,
        maxiter=None,
    ):
        """Hessian matrix of the lensing potential.

        This routine returns second derivatives computed from convergence and shear. The
        shear uses the EPL term and the inner correction term involving the series
        \\(S_2\\) (Du et al. 2020, Eq. 23-26).

        :param x: x-coordinate in image plane relative to center (major axis frame)
        :param y: y-coordinate in image plane relative to center (major axis frame)
        :param b: lens strength parameter
        :param a: outer slope \\(\\alpha\\)
        :param a_c: inner slope \\(\\alpha_c\\)
        :param r_c: break radius \\(r_c\\)
        :param q: axis ratio (minor/major)
        :param target_precision: convergence threshold for S2 recursion
        :param maxiter: maximum iteration cap for S2 recursion
        :return: f_xx, f_xy, f_yx, f_yy
        """

        tp, mi = self._resolve_settings(target_precision, maxiter)
        geom = self._prepare_geometry(x, y, q)

        pref = self._epl_prefactors(b=b, a=a, R_el=geom["R_el"], invZ=geom["invZ"])
        alpha1 = self._alpha1_epl_like(
            base1=pref["base1"], pow_a=pref["pow_a"], a=a, U_R=geom["U_R"]
        )
        kappa1 = self._kappa1_epl_like(pow_a=pref["pow_a"], a=a)

        gamma1conj = self._gamma1conj_epl_like(
            alpha1=alpha1, invZ=geom["invZ"], kappa1=kappa1, Z=geom["Z"], a=a
        )

        kappa2 = 0.0
        gamma2conj = 0.0
        if self._core_active(r_c=r_c, a=a, a_c=a_c):
            core = self._core_precompute(
                invZ=geom["invZ"],
                invZ2=geom["invZ2"],
                Z=geom["Z"],
                z2=geom["z2"],
                q=q,
                b=b,
                a=a,
                a_c=a_c,
                r_c=r_c,
                R_el=geom["R_el"],
                Beta_a=pref["Beta_a"],
                target_precision=tp,
                maxiter=mi,
                need_S0=False,
                need_S2=True,
            )
            kappa2 = core["kappa2"]
            gamma2conj = core["gamma2conj"]

        kappa = np.nan_to_num(kappa1 + kappa2, posinf=1e10, neginf=-1e10)

        gamma = gamma1conj + gamma2conj
        gamma_1 = np.nan_to_num(gamma.real, posinf=1e10, neginf=-1e10)
        gamma_2 = np.nan_to_num(-gamma.imag, posinf=1e10, neginf=-1e10)

        f_xx = kappa + gamma_1
        f_yy = kappa - gamma_1
        f_xy = gamma_2
        return f_xx, f_xy, f_xy, f_yy

    # ----------------- geometry / numerics helpers -----------------

    def _resolve_settings(self, target_precision, maxiter):
        """Resolve per-call numerical settings.

        :param target_precision: per-call convergence threshold (or None to use default)
        :param maxiter: per-call iteration cap (or None to use default)
        :return: target_precision, maxiter
        """

        tp = (
            self._target_precision
            if target_precision is None
            else float(target_precision)
        )
        mi = self._maxiter if maxiter is None else int(maxiter)
        return tp, mi

    @staticmethod
    def _elliptical_radius(x, y, q):
        """Product-averaged elliptical radius used by lenstronomy.

        :param x: x-coordinate in major-axis frame
        :param y: y-coordinate in major-axis frame
        :param q: axis ratio (minor/major)
        :return: \\(R_{\\rm el}=\\sqrt{q x^2+y^2/q}\\)
        """

        return np.sqrt(q * (x * x) + (y * y) / q).astype(np.float64)

    @staticmethod
    def _safe_complex(Z, x, y):
        """Replace Z=0 by a tiny complex number to avoid singular divisions.

        :param Z: complex coordinate array \\(Z=x+i y\\)
        :param x: x-coordinate array
        :param y: y-coordinate array
        :return: complex array with zeros replaced
        """

        Z_safe = Z.copy()
        mask0 = (x == 0) & (y == 0)
        if np.ndim(Z_safe) == 0:
            if bool(mask0):
                Z_safe = np.complex128(_TINY + 0j)
        else:
            Z_safe[mask0] = _TINY + 0j
        return Z_safe

    @staticmethod
    def _safe_nonzero_complex(arr):
        """Replace exact zeros by a tiny complex number to avoid division by zero.

        :param Z: complex array
        :return: complex array with zeros replaced
        """

        if np.ndim(arr) == 0:
            return (_TINY + 0j) if (np.abs(arr) == 0.0) else arr
        return np.where(np.abs(arr) == 0.0, (_TINY + 0j), arr)

    def _prepare_geometry(self, x, y, q):
        """Prepare common geometric quantities for derivatives and Hessian.

        :param x: x-coordinate array in major-axis frame
        :param y: y-coordinate array in major-axis frame
        :param q: axis ratio (minor/major)
        :return: dict with keys x, y, Z, R_el, z2, invZ, invZ2, U, U_R
        """

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        Z = (x + 1j * y).astype(np.complex128)
        Z = self._safe_complex(Z, x, y)

        R_el = self._elliptical_radius(x, y, q)
        R_el = np.maximum(R_el, _TINY)

        z2 = x * x + y * y

        invZ = 1.0 / Z
        invZ2 = invZ * invZ

        # zeta2 = (1/q - q)/Z^2
        U = (1.0 / q - q) * invZ2
        U_R = U * (R_el * R_el)

        return {
            "x": x,
            "y": y,
            "Z": Z,
            "R_el": R_el,
            "z2": z2,
            "invZ": invZ,
            "invZ2": invZ2,
            "U": U,
            "U_R": U_R,
        }

    # ----------------- main analytic building blocks -----------------

    def _epl_prefactors(self, b, a, R_el, invZ):
        """Compute common prefactors for the EPL (Elliptical Power Law) term.

        :param b: lens strength parameter
        :param a: outer 3D slope \\(\\alpha\\)
        :param R_el: elliptical radius \\(R_{\\rm el}\\)
        :param invZ: inverse complex coordinate \\(1/Z\\)
        :return: dict with keys Beta_a, pow_a, base1
        """
        Beta_a = self.Beta_func(a)

        pow_a = (b / R_el) ** (a - 1.0)
        base1 = (R_el * R_el) * invZ
        return {"Beta_a": Beta_a, "pow_a": pow_a, "base1": base1}

    @staticmethod
    def _core_active(r_c, a, a_c):
        """Whether the break/core correction is active.

        :param r_c: break radius \\(r_c\\)
        :param a: outer slope \\(\\alpha\\)
        :param a_c: inner slope \\(\\alpha_c\\)
        :return: True if r_c>0 and a != a_c
        """
        return (r_c > 0.0) and (a != a_c)

    @staticmethod
    def _kappa1_epl_like(pow_a, a):
        """EPL (Elliptical Power Law) convergence term \\(\\kappa_1\\) (Du et al. 2020,
        Eq. 13).

        :param pow_a: \\((b/R_{\\rm el})^{\\alpha-1}\\)
        :param a: outer slope \\(\\alpha\\)
        :return: \\(\\kappa_1\\)
        """

        return 0.5 * (3.0 - a) * pow_a

    @staticmethod
    def _alpha1_epl_like(base1, pow_a, a, U_R):
        """EPL (Elliptical Power Law) complex deflection term \\(\\alpha_1^*\\) (Du et
        al. 2020, Eq. 19).

        :param base1: \\(R_{\\rm el}^2/Z\\)
        :param pow_a: \\((b/R_{\\rm el})^{\\alpha-1}\\)
        :param a: outer slope \\(\\alpha\\)
        :param U_R: \\(\\zeta^2 R_{\\rm el}^2\\) with \\(\\zeta^2=(1/q-q)/Z^2\\)
        :return: complex deflection \\(\\alpha_1\\)
        """

        H_a = hyp2f1(0.5, (3.0 - a) / 2.0, (5.0 - a) / 2.0, U_R)
        return base1 * pow_a * H_a

    @staticmethod
    def _gamma1conj_epl_like(alpha1, invZ, kappa1, Z, a):
        """EPL (Elliptical Power Law) complex shear contribution \\(\\gamma_1^*\\) (Du
        et al. 2020, Eq. 24).

        :param alpha1: complex deflection term \\(\\alpha_1\\)
        :param invZ: \\(1/Z\\)
        :param kappa1: convergence term \\(\\kappa_1\\)
        :param Z: complex coordinate \\(Z=x+i y\\)
        :param a: outer 3D slope \\(\\alpha\\)
        :return: complex shear contribution \\(\\gamma_1^*\\)
        """

        return (2.0 - a) * (alpha1 * invZ) - kappa1 * (Z.conj() * invZ)

    # ----------------- core/break correction precompute -----------------

    def _core_precompute(
        self,
        invZ,
        invZ2,
        Z,
        z2,
        q,
        b,
        a,
        a_c,
        r_c,
        R_el,
        Beta_a,
        target_precision,
        maxiter,
        need_S0,
        need_S2,
    ):
        """Compute core/break correction terms for deflection and shear.

        This routine evaluates \\(\\kappa_2\\), and (optionally) \\(\\alpha_2\\) and
        \\(\\gamma_2^*\\) using Du et al. (2020, Eq. 20, 25), including the special
        functions F(a,z), S0, and S2.

        :param invZ: inverse complex coordinate \\(1/Z\\)
        :param invZ2: \\(1/Z^2\\)
        :param Z: complex coordinate \\(Z=x+i y\\)
        :param z2: \\(|z|^2=x^2+y^2\\)
        :param q: axis ratio (minor/major)
        :param b: lens strength parameter
        :param a: outer slope \\(\\alpha\\)
        :param a_c: inner slope \\(\\alpha_c\\)
        :param r_c: break radius \\(r_c\\)
        :param R_el: elliptical radius \\(R_{\\rm el}\\)
        :param Beta_a: \\(B(\\alpha)\\) coefficient
        :param target_precision: convergence threshold for S0/S2 recursion
        :param maxiter: maximum iteration cap for S0/S2 recursion
        :param need_S0: whether to compute \\(\\alpha_2\\) via S0
        :param need_S2: whether to compute \\(\\gamma_2^*\\) via S2
        :return: dict with keys among alpha2, kappa2, gamma2conj
        """

        out = {}

        # C = r_c^2 * zeta2
        C = (r_c * r_c) * ((1.0 / q - q) * invZ2)

        F_ac = self.F((3.0 - a_c) / 2.0, C)
        F_a3 = self.F((3.0 - a) / 2.0, C)

        if need_S0:
            S0_arr = self.S0(
                a,
                a_c,
                C,
                R_el,
                r_c,
                target_precision=target_precision,
                maxiter=maxiter,
            )
            pref2 = (r_c * r_c) * invZ * (3.0 - a) / Beta_a * (b / r_c) ** (a - 1.0)
            out["alpha2"] = pref2 * (
                2.0 / (3.0 - a_c) * F_ac - 2.0 / (3.0 - a) * F_a3 - S0_arr
            )

        # kappa2 is needed for gamma2 and for total kappa in Hessian
        kappa2 = self.kappa2func(b, a, a_c, r_c, R_el)
        out["kappa2"] = kappa2

        if need_S2:
            S2_arr = self.S2(
                a,
                a_c,
                C,
                R_el,
                r_c,
                target_precision=target_precision,
                maxiter=maxiter,
            )

            pref_g2 = (
                2.0 * (r_c * r_c) * invZ2 * (3.0 - a) / Beta_a * (b / r_c) ** (a - 1.0)
            )

            denom2 = q * (Z * Z) - (1.0 - q * q) * (r_c * r_c)
            denom2 = self._safe_nonzero_complex(denom2)

            out["gamma2conj"] = (
                pref_g2
                * (
                    (2.0 - a_c) / (3.0 - a_c) * F_ac
                    - (2.0 - a) / (3.0 - a) * F_a3
                    - S2_arr
                )
                - kappa2 * (q * z2 - (1.0 + q * q) * (r_c * r_c)) / denom2
            )

        return out

    # ----------------- special functions from Du+2020 -----------------

    @staticmethod
    def Beta_func(a):
        """Beta function coefficient \\(B(\\alpha)\\) defined in Du et al. (2020, Eq.
        5).

        :param a: outer slope \\(\\alpha\\)
        :return: \\(B(\\alpha)=\\mathrm{B}(1/2,(\\alpha-1)/2)\\)
        """

        return beta(1 / 2, (a - 1) / 2)

    def F(self, a, z):
        """Helper function F(a, z) used in the analytic expressions (Du et al. 2020, Eq.
        21).

        This is a special case of the generalized hypergeometric function \\({}_3F_2\\),
        rewritten in terms of \\({}_2F_1\\).

        :param a: scalar parameter (typically \\((3-\\alpha)/2\\) or
            \\((3-\\alpha_c)/2\\))
        :param z: complex argument (typically \\(C=r_c^2\\zeta^2\\))
        :return: F(a, z)
        """

        # Du+2020 Appendix: helper function F(a,z)
        if a == 0.5:
            return (spence(1 - np.sqrt(z)) - spence(1 + np.sqrt(z))) / np.sqrt(z) / 2
        return (1.0 / (1.0 - 2.0 * a)) * (
            hyp2f1(a, 1.0, a + 1.0, z) - 2.0 * a * hyp2f1(0.5, 1.0, 1.5, z)
        )

    def exhyp2f1(self, a, b, c, z):
        """Evaluate a numerically-stable form of \\({}_2F_1\\) for specific parameter
        combinations.

        When \\(c-a-b \\approx 1/2\\), apply a square-root transformation to improve
        convergence near branch cuts.

        :param a: \\({}_2F_1\\) parameter a
        :param b: \\({}_2F_1\\) parameter b
        :param c: \\({}_2F_1\\) parameter c
        :param z: complex argument
        :return: \\({}_2F_1(a,b;c;z)\\) evaluated stably
        """

        # "expanded" transformation for a specific parameter combination; otherwise plain 2F1
        if np.size(z) == 1:
            z = np.array([z])
        if np.isclose(c - a - b, 0.5):
            zt = np.sqrt(1 - z)
            fhyp = (2.0 / (1 + zt)) ** (2.0 * a) * hyp2f1(
                2.0 * a, a - b + 0.5, c, (zt - 1) / (zt + 1)
            )
        else:
            fhyp = hyp2f1(a, b, c, z)
        return fhyp

    # ----------------- accelerated series S0 / S2 -----------------

    def S0(self, a, a_c, C, R_el, r_c, target_precision, maxiter):
        """Series S0 used in the deflection correction term (Du et al. 2020, Eq. 22).

        Internally, this uses Aitken-accelerated recursion implemented in :meth:`s0arr`.

        :param a: outer slope \\(\\alpha\\)
        :param a_c: inner slope \\(\\alpha_c\\)
        :param C: complex coefficient \\(C=r_c^2\\zeta^2\\)
        :param R_el: elliptical radius \\(R_{\\rm el}\\)
        :param r_c: break radius \\(r_c\\)
        :param target_precision: convergence threshold for the recursion
        :param maxiter: maximum iteration cap for the recursion
        :return: complex series value S0 (same broadcast shape as inputs)
        """

        R_in = np.asarray(R_el)
        C_in = np.asarray(C)
        scalar_out = (R_in.shape == ()) and (C_in.shape == ())

        R_el = np.atleast_1d(R_in).astype(float, copy=False)
        C = np.atleast_1d(C_in).astype(complex, copy=False)

        result = C * 0.0
        mask = R_el < r_c
        if np.any(mask):
            zel2 = 1.0 - (R_el[mask] / r_c) ** 2
            result[mask] = self.s0arr(
                a, a_c, zel2, C[mask], target_precision, maxiter=maxiter
            )

        return result[0] if scalar_out else result

    def S2(self, a, a_c, C, R_el, r_c, target_precision, maxiter):
        """Series S2 used in the shear correction term (Du et al. 2020, Eq. 26).

        Internally, this uses Aitken-accelerated recursion implemented in :meth:`s2arr`.

        :param a: outer slope \\(\\alpha\\)
        :param a_c: inner slope \\(\\alpha_c\\)
        :param C: complex coefficient \\(C=r_c^2\\zeta^2\\)
        :param R_el: elliptical radius \\(R_{\\rm el}\\)
        :param r_c: break radius \\(r_c\\)
        :param target_precision: convergence threshold for the recursion
        :param maxiter: maximum iteration cap for the recursion
        :return: complex series value S2 (same broadcast shape as inputs)
        """

        R_in = np.asarray(R_el)
        C_in = np.asarray(C)
        scalar_out = (R_in.shape == ()) and (C_in.shape == ())

        R_el = np.atleast_1d(R_in).astype(float, copy=False)
        C = np.atleast_1d(C_in).astype(complex, copy=False)

        result = C * 0.0
        mask = R_el < r_c
        if np.any(mask):
            zel2 = 1.0 - (R_el[mask] / r_c) ** 2
            result[mask] = self.s2arr(
                a, a_c, zel2, C[mask], target_precision, maxiter=maxiter
            )

        return result[0] if scalar_out else result

    def s0arr(self, alpha, alphac, zel2, c, target_precision, maxiter=None):
        """Vectorized evaluator for the S0 series recursion.

        :param alpha: outer slope \\(\\alpha\\)
        :param alphac: inner slope \\(\\alpha_c\\)
        :param zel2: \\(\\tilde z^2 = 1 - (R_{\\rm el}/r_c)^2\\) (array-like)
        :param c: complex coefficient \\(C=r_c^2\\zeta^2\\) (array-like)
        :param target_precision: convergence threshold
        :param maxiter: maximum iteration cap (or None to use default)
        :return: complex array of S0 values
        """

        # Keep the original iteration/acceleration logic; expose precision/maxiter.
        nzc = np.size(zel2)
        if nzc == 1:
            zel2 = np.resize(zel2, 2)
            c = np.resize(c, 2)

        eps = target_precision
        maxiter = self._maxiter if maxiter is None else int(maxiter)

        c1 = c / (c - 1.0) * zel2
        c2 = np.sqrt(1 - c)

        s = 0.0 + 0.0j
        b = 3.0 / 2
        t = 1.0
        tc = 1.0
        a = alpha / 2.0
        ac = alphac / 2.0
        nstep = 0
        aks = c * 0.0 + 1.0

        while True:
            aks_tmp = aks * 1.0
            if nstep < 3:
                h = zel2**b * self.exhyp2f1(0.5, b, b + 1, c1) / b
                if nstep == 0:
                    s0 = s * 1.0
                if nstep == 1:
                    s1 = s * 1.0
                if nstep == 2:
                    s2 = s * 1.0
                    aks = s2 - (s2 - s1) ** 2.0 / ((s2 - s1) - (s1 - s0))
            else:
                h[sel] = zel2[sel] ** b * self.exhyp2f1(0.5, b, b + 1, c1[sel]) / b
                s0 = s1 * 1.0
                s1 = s2 * 1.0
                s2 = s * 1.0
                aks[sel] = s2[sel] - (s2[sel] - s1[sel]) ** 2.0 / (
                    (s2[sel] - s1[sel]) - (s1[sel] - s0[sel])
                )

            term = (tc - t) * h
            sel = np.where(abs(aks - aks_tmp) > eps)

            if len(sel[0]) == 0 and nstep > 5:
                break
            if nstep > maxiter:
                break

            s += term
            t = t * a / b
            tc = tc * ac / b
            a += 1
            ac += 1
            b += 1
            h = h * 0.0
            nstep += 1

        if nzc == 1:
            aks = aks[0]
            c2 = c2[0]
        return aks / c2

    def s2arr(self, alpha, alphac, zel2, c, target_precision, maxiter=None):
        """Vectorized evaluator for the S2 series recursion.

        :param alpha: outer slope \\(\\alpha\\)
        :param alphac: inner slope \\(\\alpha_c\\)
        :param zel2: \\(\\tilde z^2 = 1 - (R_{\\rm el}/r_c)^2\\) (array-like)
        :param c: complex coefficient \\(C=r_c^2\\zeta^2\\) (array-like)
        :param target_precision: convergence threshold
        :param maxiter: maximum iteration cap (or None to use default)
        :return: complex array of S2 values
        """

        # Keep the original iteration/acceleration logic; expose precision/maxiter.
        nzc = np.size(zel2)
        if nzc == 1:
            zel2 = np.resize(zel2, 2)
            c = np.resize(c, 2)

        eps = target_precision
        maxiter = self._maxiter if maxiter is None else int(maxiter)

        c1 = c / (c - 1.0) * zel2
        c3 = (1 - c) ** 1.5

        s = 0.0 + 0.0j
        b = 3.0 / 2
        t = 1.0
        tc = 1.0
        a = alpha / 2.0
        ac = alphac / 2.0
        nstep = 0
        aks = c * 0 + 1.0

        while True:
            aks_tmp = aks * 1.0
            if nstep < 3:
                h = zel2**b * self.exhyp2f1(0.5, b, b + 1, c1) * (b - 0.5) / b
                if nstep == 0:
                    s0 = s * 1.0
                if nstep == 1:
                    s1 = s * 1.0
                if nstep == 2:
                    s2 = s * 1.0
                    aks = s2 - (s2 - s1) ** 2.0 / ((s2 - s1) - (s1 - s0))
            else:
                h[sel] = (
                    zel2[sel] ** b
                    * self.exhyp2f1(0.5, b, b + 1, c1[sel])
                    * (b - 0.5)
                    / b
                )
                s0 = s1 * 1.0
                s1 = s2 * 1.0
                s2 = s * 1.0
                aks[sel] = s2[sel] - (s2[sel] - s1[sel]) ** 2.0 / (
                    (s2[sel] - s1[sel]) - (s1[sel] - s0[sel])
                )

            term = (tc - t) * h
            sel = np.where(abs(aks - aks_tmp) > eps)

            if len(sel[0]) == 0 and nstep > 5:
                break
            if nstep > maxiter:
                break

            s += term
            t = t * a / b
            tc = tc * ac / b
            a += 1
            ac += 1
            b += 1
            h = h * 0.0
            nstep += 1

        if nzc == 1:
            aks = aks[0]
            c3 = c3[0]
        return aks / c3

    def kappa2func(self, b, a, a_c, r_c, R_el):
        """Core/break correction to the convergence \\(\\kappa_2(R)\\) (Du et al. 2020,
        Eq. 14).

        :param b: lens strength parameter
        :param a: outer slope \\(\\alpha\\)
        :param a_c: inner slope \\(\\alpha_c\\)
        :param r_c: break radius \\(r_c\\)
        :param R_el: elliptical radius \\(R_{\\rm el}\\)
        :return: \\(\\kappa_2\\) (same shape as R_el)
        """

        if a == a_c or r_c <= 0:
            return (
                0.0
                if isinstance(R_el, (int, float))
                else np.zeros_like(R_el, dtype=float)
            )

        if isinstance(R_el, (int, float)):
            z_el = self.zel(R_el, r_c)
            if R_el < r_c:
                return (
                    -(3 - a)
                    / self.Beta_func(a)
                    * (b / r_c) ** (a - 1)
                    * z_el
                    * (
                        hyp2f1(a / 2, 1, 1.5, z_el**2)
                        - hyp2f1(a_c / 2, 1, 1.5, z_el**2)
                    )
                )
            return 0.0

        result = np.empty_like(R_el, dtype=float)
        mask = R_el < r_c
        if np.any(mask):
            z_el_array = self.zel(R_el[mask], r_c)
            result[mask] = (
                -(3 - a)
                / self.Beta_func(a)
                * (b / r_c) ** (a - 1)
                * z_el_array
                * (
                    hyp2f1(a / 2, 1, 1.5, z_el_array**2)
                    - hyp2f1(a_c / 2, 1, 1.5, z_el_array**2)
                )
            )
        result[~mask] = 0.0
        return result

    def zel(self, R_el, r_c):
        """Compute \\(\\tilde z = \\sqrt{1 - (R_{\\rm el}/r_c)^2}\\) used in Du et al. (2020).

        :param R_el: elliptical radius \\(R_{\\rm el}\\)
        :param r_c: break radius \\(r_c\\)
        :return: \\(\\tilde z\\) (same shape as R_el), with 0 for R_el >= r_c
        """

        if isinstance(R_el, (int, float)):
            if R_el < r_c:
                return np.sqrt(max(1.0 - (R_el**2) / (r_c**2), 0.0))
            return 0.0

        result = np.empty_like(R_el, dtype=float)
        mask = R_el < r_c
        if np.any(mask):
            r_el_array = R_el[mask]
            result[mask] = np.sqrt(np.maximum(1.0 - (r_el_array**2) / (r_c**2), 0.0))
        result[~mask] = 0.0
        return result

    def kappa_mean(self, R, alpha, alpha_c, b, r_c):
        """Mean convergence \\(\\bar\\kappa(R)\\) inside projected radius R.

        :param R: projected radius (scalar)
        :param alpha: outer slope \\(\\alpha\\)
        :param alpha_c: inner slope \\(\\alpha_c\\)
        :param b: lens strength parameter
        :param r_c: break radius \\(r_c\\)
        :return: mean convergence \\(\\bar\\kappa(R)\\)
        """

        R = float(R)
        temp1 = (b / R) ** (alpha - 1)
        if R < r_c:
            z = np.sqrt(max(1.0 - (R / r_c) ** 2, 0.0))
            temp3 = (
                2.0
                / 3.0
                * (3 - alpha)
                / self.Beta_func(alpha)
                * (b / r_c) ** (alpha - 1)
                * (r_c / R) ** 2
                * z**3
                * (hyp2f1(alpha / 2, 1, 2.5, z**2) - hyp2f1(alpha_c / 2, 1, 2.5, z**2))
            )
            temp4 = (
                -2.0
                / 3.0
                * (3 - alpha)
                / self.Beta_func(alpha)
                * (b / r_c) ** (alpha - 1)
                * (r_c / R) ** 2
                * (hyp2f1(alpha / 2, 1, 2.5, 1.0) - hyp2f1(alpha_c / 2, 1, 2.5, 1.0))
            )
            return temp1 + temp3 + temp4

        temp4 = (
            -2.0
            / 3.0
            * (3 - alpha)
            / self.Beta_func(alpha)
            * (b / r_c) ** (alpha - 1)
            * (r_c / R) ** 2
            * (hyp2f1(alpha / 2, 1, 2.5, 1.0) - hyp2f1(alpha_c / 2, 1, 2.5, 1.0))
        )
        return temp1 + temp4

    def phi_r(self, xi, alpha, alpha_c, b, r_c):
        """Radial integrand kappa_mean(xi) * xi used in the potential quadrature.

        :param xi: projected radius xi
        :param alpha: outer slope \\(\\alpha\\)
        :param alpha_c: inner slope \\(\\alpha_c\\)
        :param b: lens strength parameter
        :param r_c: break radius \\(r_c\\)
        :return: kappa_mean(xi) * xi
        """

        return self.kappa_mean(xi, alpha, alpha_c, b, r_c) * xi

    def integrand_psi(self, u, x, y, alpha, alpha_c, b, r_c, q):
        """Integrand for the 1D potential quadrature.

        :param u: integration variable in (0, 1)
        :param x: x-coordinate in major-axis frame
        :param y: y-coordinate in major-axis frame
        :param alpha: outer slope \\(\\alpha\\)
        :param alpha_c: inner slope \\(\\alpha_c\\)
        :param b: lens strength parameter
        :param r_c: break radius \\(r_c\\)
        :param q: axis ratio (minor/major)
        :return: integrand value for the potential \\(\\psi\\)
        """

        xi = np.sqrt(u * q * (x**2 + y**2 / (1 - (1 - q**2) * u)))
        return (
            0.5
            * (xi / u)
            * self.phi_r(xi, alpha, alpha_c, b, r_c)
            / np.sqrt(1 - (1 - q**2) * u)
        )
