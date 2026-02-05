__author__ = "Guanhua Rui,  Wei Du"

#  this file contains a class to make a Power-Law Sersic profile

import numpy as np

# from lenstronomy.LensModel.Profiles.sersic_utils import SersicUtil
from lenstronomy.Util.package_util import exporter
from lenstronomy.Util import param_util
from scipy.special import gamma, hyp2f1, gammaincc
from scipy.integrate import quad

export, __all__ = exporter()


@export
class PL_Sersic(object):
    """This class contains functions to evaluate a 2D PL-Sérsic surface brightness
    profile.

    The surface luminosity density profile corresponding to the 3D PL-Sérsic profile is written as (see Wei, Du 2020)):

    - For :math:`R < r_c`, :math:`I(R) = 2 amp r_c \\tilde{z} 2F1(alpha_c/2, 1; 3/2; \\tilde{z}^2) + 2 \int_{r_c}^{\infty} j(r) r dr / sqrt(r^2 - R^2)`
    - For :math:`R > r_c`, :math:`I(R) = I_0 \\exp{-(R/s)^{nu}}`

    with :math:`j_c = amp`
    """

    param_names = [
        "amp",
        "R_sersic",
        "n_sersic",
        "alpha_c",
        "r_c",
        "e1",
        "e2",
        "center_x",
        "center_y",
    ]

    lower_limit_default = {
        "amp": 0,
        "R_sersic": 0,
        "n_sersic": 0.5,
        "alpha_c": 0.1,
        "r_c": 0.1,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
    }

    upper_limit_default = {
        "amp": 100,
        "R_sersic": 100,
        "n_sersic": 8,
        "alpha_c": 3.0,
        "r_c": 10,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
    }

    def calculate_u_nu(self, n_sersic):
        """Calculate u and nu based on n_sersic.

        :param n_sersic: Sérsic index
        :return: (u, nu) parameters
        """
        nu = 1 / n_sersic
        u = 1 - 0.6097 * nu + 0.054635 * nu**2
        return u, nu

    def calculate_param(self, R_sersic, n_sersic):
        """Calculate the scale radius s from the 2D effective radius R_sersic and the
        Sérsic index n.

        :param R_sersic: 2D effective radius
        :param n_sersic: Sérsic index
        :return: scale radius s
        """
        u, nu = self.calculate_u_nu(n_sersic)
        k = (2 * n_sersic) ** (
            1 / n_sersic
        )  # Based on the relationship for k from the literature
        s = R_sersic / k**n_sersic
        return s, u, nu

    def get_distance_from_center_bpl(self, x, y, e1, e2, center_x, center_y):
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)
        xt = cos_phi * x_shift + sin_phi * y_shift
        yt = -sin_phi * x_shift + cos_phi * y_shift
        R = np.sqrt(xt * xt * q + yt * yt / q)
        return R

    def function(
        self,
        x,
        y,
        amp,
        R_sersic,
        n_sersic,
        alpha_c,
        r_c,
        e1,
        e2,
        center_x=0,
        center_y=0,
        max_R_frac=1000.0,
    ):
        """PL-Sérsic 2D surface brightness from a 3D PL-Sérsic luminosity density:

        I(R) = 2 amp r_c \\tilde{z} 2F1(alpha_c/2, 1; 3/2; \\tilde{z}^2)
               + 2 \int_{r_c}^{\infty} j(r) r dr / sqrt(r^2 - R^2)     (R < r_c)
        I(R) = I_0 exp[-(R/s)^nu]                                      (R >= r_c)
        """
        R = self.get_distance_from_center_bpl(x, y, e1, e2, center_x, center_y)

        # cutoff
        R_max = max_R_frac * R_sersic
        s, u, nu = self.calculate_param(R_sersic, n_sersic)

        # --- outer 3D PL-Sérsic density j(r) for r >= r_c ---
        # assume: j(r) = j0 * (r/s)^(-u) * exp[-(r/s)^nu]
        # enforce continuity at r = r_c: j(r_c) = amp
        rcvs = r_c / s
        j0 = amp * (rcvs**u) * np.exp(rcvs**nu)

        def j_outer(ri):
            rvs = ri / s
            return j0 * (rvs ** (-u)) * np.exp(-(rvs**nu))

        # I0 for the outer projected approximation
        I_0 = 2.0 * s * j0 * (gamma((3.0 - u) / nu) / gamma(2.0 / nu))

        def I_inner_scalar(Ri):
            # \tilde{z} = sqrt(1 - (R/r_c)^2)
            zt = np.sqrt(1.0 - (Ri / r_c) ** 2)
            term_analytic = (
                2.0 * amp * r_c * zt * hyp2f1(alpha_c / 2.0, 1.0, 1.5, zt**2)
            )

            # 2 * \int_{r_c}^{\infty} j(r) r / sqrt(r^2 - R^2) dr
            integral, _ = quad(
                lambda ri: j_outer(ri) * ri / np.sqrt(ri * ri - Ri * Ri),
                r_c,
                np.inf,
                limit=200,
            )
            return term_analytic + 2.0 * integral

        # ---- array handling ----
        if np.ndim(R) == 0:
            if R > R_max:
                return 0.0
            # put R == r_c into the outer branch for numerical stability
            if R < r_c:
                return I_inner_scalar(float(R))
            return I_0 * np.exp(-((R / s) ** nu))

        # array case
        I_array = np.zeros_like(R, dtype=float)
        valid = R <= R_max

        outer_mask = valid & (R >= r_c)
        I_array[outer_mask] = I_0 * np.exp(-((R[outer_mask] / s) ** nu))

        inner_mask = valid & (R < r_c)
        if np.any(inner_mask):
            # loop only over inner pixels (usually few); integral is expensive
            idxs = np.argwhere(inner_mask)
            for idx in idxs:
                idx = tuple(idx)
                I_array[idx] = I_inner_scalar(float(R[idx]))

        return I_array

    def _s_u_nu_j0(self, amp, R_sersic, n_sersic, r_c):
        """Return (s, u, nu, j0, x_c) consistent with the definitions in function().

        x_c = (r_c/s)^nu
        """
        s, u, nu = self.calculate_param(R_sersic, n_sersic)
        rcvs = r_c / s
        x_c = rcvs**nu
        # continuity: j(r_c) = amp
        j0 = amp * (rcvs**u) * np.exp(x_c)
        return s, u, nu, j0, x_c

    def total_flux(self, amp, R_sersic, n_sersic, alpha_c, r_c, e1=0, e2=0, **kwargs):
        """Analytic total flux from the *3D* luminosity density j(r):

            F_tot = 4*pi * [ amp*r_c^3/(3-alpha_c)
                            + j0*s^3/nu * Gamma((3-u)/nu, (r_c/s)^nu) ]

        where Gamma(a,x) is the upper incomplete gamma function.
        """
        if alpha_c >= 3.0:
            raise ValueError(
                "alpha_c must be < 3 for finite total flux in the inner power-law part."
            )

        s, u, nu, j0, x_c = self._s_u_nu_j0(amp, R_sersic, n_sersic, r_c)

        a = (3.0 - u) / nu
        # upper incomplete gamma: Γ(a,x) = Γ(a) * gammaincc(a,x)
        Gamma_upper = gamma(a) * gammaincc(a, x_c)

        inner = amp * r_c**3 / (3.0 - alpha_c)
        outer = j0 * s**3 / nu * Gamma_upper
        return 4.0 * np.pi * (inner + outer)
