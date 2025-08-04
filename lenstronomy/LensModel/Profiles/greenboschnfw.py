__author__ = "jtekverk"

import numpy as np
from scipy.integrate import quad
from lenstronomy.LensModel.Profiles.radial_interpolated import RadialInterpolate
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.Cosmo.lens_cosmo import LensCosmo

__all__ = ["GreenBoschNFW"]
model_name = "GreenBoschNFW"

class GreenBoschNFW(LensProfileBase):

    param_names = [
        'f_b',      # Bound mass fraction M_bound / M_infall
        'c_s',      # Infall concentration 
        'Rs',       # Infall scale radius [arcseconds]
        'rho0',     # NFW density profile normalization [M_solar / Mpc^3]
        'center_x', # x-coordinate in lens plane [arcseconds]
        'center_y'  # y-coordinate in lens plane [arcseconds]
    ]
    lower_limit_default = {
        "f_b": 1.0e-5,
        "c_s": 0.02,
        "Rs": 0.02,
        "rho0": 0.0,
        "center_x": -10000.0,
        "center_y": -10000.0
    }
    upper_limit_default = {
        "f_b": 1.0,
        "c_s":    1000.0,
        "Rs":    1000.0,
        "rho0":  1.0e25,
        "center_x": 10000.0,
        "center_y": 10000.0
    }

    def __init__(self, z_lens, z_source, cosmo, **kwargs_numerics):

        self._last_params = None
        self._cached = None
        self.lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo)
        self.sigma_crit = self.lens_cosmo.sigma_crit
        super().__init__()
        self._rad_interp = RadialInterpolate(**kwargs_numerics)

    def function(self, x, y, f_b, c_s, Rs, rho0, center_x, center_y):

        kappa_r, r_bin = self.rbin_kappa_r(f_b, c_s, Rs, rho0)

        return self._rad_interp.function(
            x, y,
            r_bin=r_bin,
            kappa_r=kappa_r,
            center_x=center_x,
            center_y=center_y
        )

    def derivatives(self, x, y, f_b, c_s, Rs, rho0, center_x, center_y):

        kappa_r, r_bin = self.rbin_kappa_r(f_b, c_s, Rs, rho0)

        return self._rad_interp.derivatives(
            x, y,
            r_bin=r_bin,
            kappa_r=kappa_r,
            center_x=center_x,
            center_y=center_y
        )
    
    def hessian(self, x, y, f_b, c_s, Rs, rho0, center_x, center_y):

        kappa_r, r_bin = self.rbin_kappa_r(f_b, c_s, Rs, rho0)

        return self._rad_interp.hessian(
            x, y,
            r_bin=r_bin,
            kappa_r=kappa_r,
            center_x=center_x,
            center_y=center_y
        )
    
    def set_dynamic(self):

        self._last_params = None
        self._cached = None
        self._rad_interp.set_dynamic()


    def rho_3d_lens(self, r, f_b, c_s, Rs, rho0):
    
        a1, a2, a3, a4 = 0.338, 0.0, 0.157, 1.337
        b1, b2, b3, b4, b5, b6 = 0.448, 0.272, -0.199, 0.011, -1.119, 0.093
        c0, c1, c2, c3, c4 = 2.779, -0.035, -0.337, -0.099, 0.415
        f_te = f_b**(a1*(c_s/10.0)**a2) * c_s**(a3*(1.0-f_b)**a4)
        r_te = c_s*f_b**(b1 * (c_s/10.0)**b2) * c_s**(b3 * (1.0-f_b)**b4) * np.exp(b5 * (c_s/10.0)**b6 * (1.0-f_b))
        delta = c0*f_b**(c1 * (c_s/10.0)**c2) * c_s**(c3 * (1.0-f_b)**c4)
        coeff = (c_s - r_te) / (c_s * r_te)

        return (f_te*rho0) / ( (1+(coeff * r/Rs)**delta)*(r/Rs)*(1+r/Rs)**2 )
    
    def rbin_kappa_r(self, f_b, c_s, Rs, rho0):
        
        def _round_params(p):
            return tuple(np.round(p, decimals=10))

        params = _round_params((f_b, c_s, Rs, rho0))

        if self._last_params == params and self._cached is not None:
            return self._cached

        r_min, r_max, num_bins = 1.0e-4, 10.0*Rs, 100
        r_bin = np.logspace(np.log10(r_min), np.log10(r_max), num_bins)
        kappa_vals = []

        for r in r_bin:

            integrand = lambda z: 2.0*self.rho_3d_lens(np.hypot(r,z), f_b, c_s, Rs, rho0)
            sigma, error = quad(integrand, 0, np.inf, limit=200)

            if not np.isfinite(sigma) or not np.isfinite(error):
                raise RuntimeError(f"LOS integral failed at r={r}")
            
            kappa = sigma / self.sigma_crit
            kappa_vals.append(kappa)
            
            if sigma != 0:
              rel_err = abs(error / sigma)
            else:
              rel_err = np.inf
            if rel_err > 1e-3:
               print(f"High relative LOS integral error at r={r:.3e}: rel_err={rel_err:.2e}")

        kappa_r = np.array(kappa_vals)
        self._last_params = params
        self._cached = (kappa_r, r_bin)

        return kappa_r, r_bin

