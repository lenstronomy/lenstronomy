__author__ = "Guanhua Rui, Wei Du"

import lenstronomy.Util.util as util
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.spp import SPP
import numpy as np
from scipy.integrate import quad
from scipy.special import hyp2f1, spence, beta
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

__all__ = ["BPL", "BPLMajorAxis"]

_TINY = 1e-15


class BPL(LensProfileBase):
    """Broken Power Law mass profile.

    The mathematical form follows Wei, Du (2020).
    """

    # b, a, a_c, r_c,q
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
        self.bpl_major_axis = BPLMajorAxis()
        super(BPL, self).__init__()

    def param_conv(self, b, a, a_c, r_c, e1, e2):
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
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        return b, a, a_c, r_c, q, phi_G

    def set_static(self, b, a, a_c, r_c, e1, e2, center_x=0, center_y=0):
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

    def function(self, x, y, b, a, a_c, r_c, e1, e2, center_x=0, center_y=0):
        b, a, a_c, r_c, q, phi_G = self.param_conv(b, a, a_c, r_c, e1, e2)
        x_ = x - center_x
        y_ = y - center_y
        x__, y__ = util.rotate(x_, y_, phi_G)
        f_ = self.bpl_major_axis.function(x__, y__, b, a, a_c, r_c, q)
        return f_

    def derivatives(self, x, y, b, a, a_c, r_c, e1, e2, center_x=0, center_y=0):
        b, a, a_c, r_c, q, phi_G = self.param_conv(b, a, a_c, r_c, e1, e2)
        x_ = x - center_x
        y_ = y - center_y
        x__, y__ = util.rotate(x_, y_, phi_G)
        f__x, f__y = self.bpl_major_axis.derivatives(x__, y__, b, a, a_c, r_c, q)
        f_x, f_y = util.rotate(f__x, f__y, -phi_G)
        return f_x, f_y

    def hessian(self, x, y, b, a, a_c, r_c, e1, e2, center_x=0, center_y=0):
        b, a, a_c, r_c, q, phi_G = self.param_conv(b, a, a_c, r_c, e1, e2)
        x_ = x - center_x
        y_ = y - center_y
        x__, y__ = util.rotate(x_, y_, phi_G)
        f__xx, f__xy, f__yx, f__yy = self.bpl_major_axis.hessian(
            x__, y__, b, a, a_c, r_c, q
        )
        kappa = 0.5 * (f__xx + f__yy)
        gamma1__ = 0.5 * (f__xx - f__yy)
        gamma2__ = f__xy

        # ---- enforce well-defined shear at the exact lens center ----
        mask0 = (np.asarray(x_) == 0) & (np.asarray(y_) == 0)
        if np.ndim(mask0) == 0:
            if bool(mask0):
                gamma1__ = 0.0
                gamma2__ = 0.0
        else:
            if np.any(mask0):
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

    def Beta_func(self, a):
        return beta(1 / 2, (a - 1) / 2)

    def rho_c_from_b(self, b, a, r_c):
        """Compute ρ_c from (b, a, r_c) using Du+2020 Eq. (8), in lens units.

        b^(a-1) = B(a) * 2/(3-a) * ρ_c * r_c^a

        => ρ_c = (3-a) / (2 * B(a)) * b^(a-1) / r_c^a
        """
        B_a = self.Beta_func(a)
        rho_c = (3.0 - a) * b ** (a - 1.0) / (2.0 * B_a * r_c**a)
        return rho_c

    def mass_3d_lens(self, r, b, a, a_c, r_c, e1=None, e2=None):
        """3D enclosed mass M(<r) for the BPL model, using lens parameters.

        Inputs
        ------
        r         : float or array, 3D radius
        b         : scale radius b (Du+2020,2023)
        a         : outer 3D slope α
        a_c       : inner 3D slope α_c
        r_c       : break radius r_c

        Returns
        -------
        M : float or array, 3D mass inside radius r
        """
        r = np.asarray(r, dtype=float)

        # from (b, a, r_c) to ρ_c
        rho_c = self.rho_c_from_b(b, a, r_c)

        # allocate output
        M = np.zeros_like(r, dtype=float)

        # sanity checks
        if np.isclose(3.0 - a_c, 0.0):
            raise ValueError("a_c = 3 causes divergent inner mass.")
        if np.isclose(3.0 - a, 0.0):
            raise ValueError("a = 3 causes divergent outer mass.")

        # masks
        inner = r <= r_c
        outer = ~inner

        # r <= r_c : inner slope α_c
        if np.any(inner):
            r_in = r[inner]
            M[inner] = (
                4.0 * np.pi * rho_c * r_c**a_c * r_in ** (3.0 - a_c) / (3.0 - a_c)
            )

        # r >= r_c : outer slope α, plus constant m0
        if np.any(outer):
            r_out = r[outer]

            # Du+2020 Eq. (3):
            # m0 = - 4π ρ_c / (3-α) * (α - α_c)/(3-α_c) * r_c^3
            m0 = -4.0 * np.pi * rho_c * (a - a_c) * r_c**3 / ((3.0 - a) * (3.0 - a_c))

            M[outer] = (
                4.0 * np.pi * rho_c * r_c**a * r_out ** (3.0 - a) / (3.0 - a) + m0
            )

        return M

    # TODO
    # def density_lens(self, r, b, a, a_c, r_c, e1=None, e2=None):
    #     return self.density_lens(r, b, a, a_c, r_c)


class BPLMajorAxis(LensProfileBase):
    """Major-axis-aligned implementation of BPL."""

    param_names = ["b", "a", "a_c", "r_c", "center_x", "center_y"]

    def __init__(self):
        super(BPLMajorAxis, self).__init__()

    def function(self, x, y, b, a, a_c, r_c, q):
        vectorized_quad = np.vectorize(
            lambda aa, bb: quad(
                self.integrand_psi, 0, 1, args=(aa, bb, a, a_c, b, r_c, q)
            )[0]
        )
        psi = vectorized_quad(x, y)
        return psi

    # ------------- derivatives / hessian -------------

    def derivatives(self, x, y, b, a, a_c, r_c, q):
        """Returns the deflection angles (alpha_x, alpha_y) with precomputation &
        reuse."""

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        Z = (x + 1j * y).astype(np.complex128)

        Z_safe = Z.copy()
        mask0 = (x == 0) & (y == 0)
        if np.ndim(Z_safe) == 0:
            if bool(mask0):
                Z_safe = _TINY + 0j
        else:
            Z_safe[mask0] = _TINY + 0j

        R_el = np.sqrt(x * x * q + y * y / q).astype(np.float64)
        R_el_safe = np.maximum(R_el, _TINY)

        invZ = 1.0 / Z_safe
        invZ2 = invZ * invZ
        Beta_a = self.Beta_func(a)
        b_over_R = b / R_el_safe
        pow_a = b_over_R ** (a - 1.0)
        base1 = (R_el_safe * R_el_safe) * invZ

        # zeta2 = (1/q - q)/Z^2
        U = (1.0 / q - q) * invZ2
        U_R = U * (R_el_safe * R_el_safe)

        #
        H_a = hyp2f1(0.5, (3.0 - a) / 2.0, (5.0 - a) / 2.0, U_R)

        # alpha1（EPL-like ）
        alpha1 = base1 * pow_a * H_a

        # alpha2（core）
        if r_c != 0 and a != a_c:
            C = (r_c * r_c) * U
            F_ac = self.F((3.0 - a_c) / 2.0, C)
            F_a3 = self.F((3.0 - a) / 2.0, C)
            S0_arr = self.S0(a, a_c, C, R_el_safe, r_c, target_precision=1e-6)

            pref2 = (r_c * r_c) * invZ * (3.0 - a) / Beta_a * (b / r_c) ** (a - 1.0)
            alpha2 = pref2 * (
                2.0 / (3.0 - a_c) * F_ac - 2.0 / (3.0 - a) * F_a3 - S0_arr
            )
        else:
            alpha2 = 0.0

        alpha = alpha1 + alpha2

        alpha_real = np.nan_to_num(alpha.real, posinf=1e15, neginf=-1e15)
        alpha_imag = np.nan_to_num(-alpha.imag, posinf=1e15, neginf=-1e15)
        return alpha_real, alpha_imag

    def hessian(self, x, y, b, a, a_c, r_c, q):
        """Hessian matrix (f_xx, f_xy, f_yx, f_yy) with precomputation & reuse."""

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        Z = (x + 1j * y).astype(np.complex128)

        Z_safe = Z.copy()
        mask0 = (x == 0) & (y == 0)
        if np.ndim(Z_safe) == 0:
            if bool(mask0):
                Z_safe = _TINY + 0j
        else:
            Z_safe[mask0] = _TINY + 0j

        R_el = np.sqrt(x * x * q + y * y / q).astype(np.float64)
        R_el_safe = np.maximum(R_el, _TINY)

        z2 = x * x + y * y  # |z|^2

        invZ = 1.0 / Z_safe
        invZ2 = invZ * invZ
        Beta_a = self.Beta_func(a)
        b_over_R = b / R_el_safe
        pow_a = b_over_R ** (a - 1.0)
        base1 = (R_el_safe * R_el_safe) * invZ

        # zeta2
        U = (1.0 / q - q) * invZ2
        U_R = U * (R_el_safe * R_el_safe)

        # kappa1 / kappa2
        kappa1 = (3.0 - a) * 0.5 * pow_a
        kappa2 = self.kappa2func(b, a, a_c, r_c, R_el_safe)
        kappa = np.nan_to_num(kappa1 + kappa2, posinf=1e10, neginf=-1e10)

        H_a = hyp2f1(0.5, (3.0 - a) / 2.0, (5.0 - a) / 2.0, U_R)
        C = (r_c * r_c) * U
        F_ac = self.F((3.0 - a_c) / 2.0, C)
        F_a3 = self.F((3.0 - a) / 2.0, C)
        S2_arr = self.S2(a, a_c, C, R_el_safe, r_c, target_precision=1e-6)

        # alpha1（ gamma1conj ）
        alpha1 = base1 * pow_a * H_a

        # gamma1conj = (2-a)*alpha1/Z - kappa1 * Z* / Z
        gamma1conj = (2.0 - a) * (alpha1 * invZ) - kappa1 * (Z_safe.conj() * invZ)

        # gamma2conj（core + kappa2 ）
        pref_g2 = (
            2.0 * (r_c * r_c) * invZ2 * (3.0 - a) / Beta_a * (b / r_c) ** (a - 1.0)
        )
        #  q*Z^2 - (1+q^2)*r_c^2
        denom2 = q * (Z_safe * Z_safe) - (1.0 + q * q) * (r_c * r_c)
        if np.ndim(denom2) == 0:
            if np.abs(denom2) == 0.0:
                denom2 = _TINY + 0j
        else:
            denom2 = np.where(np.abs(denom2) == 0.0, (_TINY + 0j), denom2)

        gamma2conj = (
            pref_g2
            * ((2.0 - a_c) / (3.0 - a_c) * F_ac - (2.0 - a) / (3.0 - a) * F_a3 - S2_arr)
            - kappa2 * (q * z2 - (1.0 + q * q) * (r_c * r_c)) / denom2
        )

        gamma = gamma1conj + gamma2conj

        gamma_1 = np.nan_to_num(gamma.real, posinf=1e10, neginf=-1e10)
        gamma_2 = np.nan_to_num(-gamma.imag, posinf=1e10, neginf=-1e10)

        f_xx = kappa + gamma_1
        f_yy = kappa - gamma_1
        f_xy = gamma_2
        return f_xx, f_xy, f_xy, f_yy

    # --------------------------------------------------------------------------

    def Beta_func(self, a):
        return beta(1 / 2, (a - 1) / 2)

    def F(self, a, z):
        if a == 0.5:
            return (spence(1 - np.sqrt(z)) - spence(1 + np.sqrt(z))) / np.sqrt(z) / 2
        else:
            return (1 / (1 - 2 * a)) * (
                hyp2f1(a, 1, a + 1, z) - 2 * a * hyp2f1(0.5, 1, 1.5, z)
            )

    def exhyp2f1(self, a, b, c, z):
        if np.size(z) == 1:
            z = np.array([z])
        if c - a - b == 0.5:
            zt = np.sqrt(1 - z)
            fhyp = (2.0 / (1 + zt)) ** (2.0 * a) * hyp2f1(
                2.0 * a, a - b + 0.5, c, (zt - 1) / (zt + 1)
            )
        else:
            fhyp = hyp2f1(a, b, c, z)
        return fhyp

    def S0(self, a, a_c, C, R_el, r_c, target_precision):
        if isinstance(R_el, (int, float)):
            R_el = np.array([R_el])
        result = C * 0
        sel = np.where(R_el < r_c)
        if len(sel[0]) > 0 and sel[0][0] != 0:
            zel2 = 1.0 - (R_el[sel] / r_c) ** 2
            cc = C[sel]
            result[sel] = self.s0arr(a, a_c, zel2, cc, target_precision)
        return result

    def s0arr(self, alpha, alphac, zel2, c, target_precision):
        nzc = np.size(zel2)
        if nzc == 1:
            zel2 = np.resize(zel2, 2)
            c = np.resize(c, 2)
        eps = target_precision
        maxiter = 300
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
        while 1:
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

    def s2arr(self, alpha, alphac, zel2, c, target_precision):
        nzc = np.size(zel2)
        if nzc == 1:
            zel2 = np.resize(zel2, 2)
            c = np.resize(c, 2)
        eps = target_precision
        maxiter = 300
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
        while 1:
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

    def S2(self, a, a_c, C, R_el, r_c, target_precision):
        if isinstance(R_el, (int, float)):
            R_el = np.array([R_el])
        result = C * 0
        sel = np.where(R_el < r_c)
        if len(sel[0]) > 0 and sel[0][0] != 0:
            zel2 = 1.0 - (R_el[sel] / r_c) ** 2
            cc = C[sel]
            result[sel] = self.s2arr(a, a_c, zel2, cc, target_precision)
        return result

    def kappa2func(self, b, a, a_c, r_c, R_el):
        if a == a_c:
            # exact power-law limit: the core/break correction vanishes identically
            return (
                0.0
                if isinstance(R_el, (int, float))
                else np.zeros_like(R_el, dtype=np.float64)
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
            else:
                return 0.0
        else:
            result = np.empty_like(R_el, dtype=np.float64)
            mask = R_el < r_c
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
        if isinstance(R_el, (int, float)):
            if R_el < r_c:
                return np.sqrt(max(1.0 - (R_el**2) / (r_c**2), 0.0))
            else:
                return 0.0
        else:
            result = np.empty_like(R_el, dtype=np.float64)
            mask = R_el < r_c
            r_el_array = R_el[mask]
            result[mask] = np.sqrt(np.maximum(1.0 - (r_el_array**2) / (r_c**2), 0.0))
            result[~mask] = 0.0
            return result

    def kappa_mean(self, R, alpha, alpha_c, b, r_c):
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
                * 1.0**3
                * (hyp2f1(alpha / 2, 1, 2.5, 1.0) - hyp2f1(alpha_c / 2, 1, 2.5, 1.0))
            )
            return temp1 + temp3 + temp4
        else:
            temp4 = (
                -2.0
                / 3.0
                * (3 - alpha)
                / self.Beta_func(alpha)
                * (b / r_c) ** (alpha - 1)
                * (r_c / R) ** 2
                * 1.0**3
                * (hyp2f1(alpha / 2, 1, 2.5, 1.0) - hyp2f1(alpha_c / 2, 1, 2.5, 1.0))
            )
            return temp1 + temp4

    def phi_r(self, xi, alpha, alpha_c, b, r_c):
        return self.kappa_mean(xi, alpha, alpha_c, b, r_c) * xi

    def integrand_psi(self, u, x, y, alpha, alpha_c, b, r_c, q):
        xi = np.sqrt(u * q * (x**2 + y**2 / (1 - (1 - q**2) * u)))
        return (
            0.5
            * (xi / u)
            * self.phi_r(xi, alpha, alpha_c, b, r_c)
            / np.sqrt(1 - (1 - q**2) * u)
        )
