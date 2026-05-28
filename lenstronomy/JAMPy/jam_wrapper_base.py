__author__ = "furcelay", "sibirrer"

from lenstronomy.GalKin.cosmo import Cosmo
from lenstronomy.GalKin.anisotropy import Anisotropy
from lenstronomy.Util.param_util import ellipticity2phi_q
import jampy as jam
import numpy as np

__all__ = ["JAMWrapperBase"]


class JAMWrapperBase(object):
    """Wrapper class to use jampy JAM functionality similar to lenstronomy's Galkin
    class.

    :param kwargs_model: keyword arguments describing the model components
    :param kwargs_cosmo: keyword arguments that define the cosmology in terms of the
        angular diameter distances involved
    """

    def __init__(
        self,
        kwargs_model,
        kwargs_cosmo,
    ):
        self.mass_profile_list = kwargs_model.get("mass_profile_list")
        self.light_profile_list = kwargs_model.get("light_profile_list")
        if (len(self.mass_profile_list) > 1) or (
            self.mass_profile_list[0]
            not in ["MULTI_GAUSSIAN", "MULTI_GAUSSIAN_ELLIPSE_KAPPA"]
        ):
            raise ValueError(
                "Jampy only support MULTI_GAUSSIAN(_ELLIPSE) mass profiles"
            )
        if (len(self.light_profile_list) > 1) or (
            self.light_profile_list[0]
            not in ["MULTI_GAUSSIAN", "MULTI_GAUSSIAN_ELLIPSE"]
        ):
            raise ValueError(
                "Jampy only support MULTI_GAUSSIAN(_ELLIPSE) light profiles"
            )
        anisotropy_model = kwargs_model.get("anisotropy_model")
        self._anisotropy = Anisotropy(anisotropy_model)
        self.cosmo = Cosmo(**kwargs_cosmo)

        self.symmetry = kwargs_model.get(
            "symmetry", "spherical"
        )  # 'spherical', 'axi_sph', 'axi_cyl'
        self.align = None
        self.axisymmetric = False
        if self.symmetry == "spherical":
            self.align = "sph"
            self.axisymmetric = False
        elif self.symmetry == "axi_sph":
            self.axisymmetric = True
            self.align = "sph"
        elif self.symmetry == "axi_cyl":
            self.axisymmetric = True
            self.align = "cyl"
        else:
            msg = (
                f"Invalid symmetry type '{self.symmetry}' for JAMWrapper, "
                f"options are 'spherical', 'axi_sph' or 'axi_cyl'."
            )
            raise ValueError(msg)

        self.cosmo = Cosmo(**kwargs_cosmo)

    def dispersion_points(
        self,
        x,
        y,
        kwargs_mass,
        kwargs_light,
        kwargs_anisotropy,
        inclination=90.0,
        convolved=False,
        psf_sigmas=0,
        psf_amplitudes=1,
        delta_pix=0,
        black_hole_mass=0,
        kwargs_jampy=None,
    ):
        """Computes the LOS velocity dispersion at given points (not convolved).

        :param x: array of x positions where to compute the dispersion [arcsec]
        :param y: array of y positions where to compute the dispersion [arcsec]
        :param kwargs_mass: mass model parameters (following lenstronomy lens model
            conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light
            model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to
            anisotropy type chosen.
        :param inclination: inclination angle of the system [degrees]
        :param convolved: bool, if True the PSF convolution is applied
        :param psf_sigmas: float or array with PSF gaussian sigmas [arcsec]
        :param psf_amplitudes: 1 or array with PSF amplitudes
        :param delta_pix: delta pix
        :param black_hole_mass: mass of the central SMBH [solar masses]
        :param kwargs_jampy: keyword arguments for JAM call
        :return: array of LOS velocity dispersion at each (x,y) position [km/s]
        """

        sigma_lum = np.asarray(kwargs_light[0]["sigma"])
        # convert to surface brightness
        surf_lum = np.asarray(kwargs_light[0]["amp"]) / (2 * np.pi * sigma_lum**2)

        sigma_mass = np.asarray(kwargs_mass[0]["sigma"])
        # convert to convergence
        surf_mass = np.asarray(kwargs_mass[0]["amp"]) / (2 * np.pi * sigma_mass**2)
        # convert to units of M_sun / pc^2
        surf_mass *= self.cosmo.epsilon_crit * 1e-12

        beta = self._anisotropy.jampy_params(kwargs_anisotropy)
        if not self._anisotropy.use_logistic:
            beta = beta * np.ones_like(surf_lum)
        _, q_lum = ellipticity2phi_q(*self._extract_ellipticity(kwargs_light))
        _, q_mass = ellipticity2phi_q(*self._extract_ellipticity(kwargs_mass))
        if not convolved:
            psf_sigmas = 0.0
            delta_pix = 0.0
            psf_amplitudes = 1.0

        vrms, surf_bright = self.call_jampy(
            surf_lum,
            sigma_lum,
            surf_mass,
            sigma_mass,
            x=x,
            y=y,
            q_lum=q_lum * np.ones_like(surf_lum),
            q_mass=q_mass * np.ones_like(surf_mass),
            inclination=inclination,
            beta=beta,
            sigma_psf=psf_sigmas,
            norm_psf=psf_amplitudes,
            pix_size=delta_pix,
            black_hole_mass=black_hole_mass,
            jam_kwargs=kwargs_jampy,
        )
        return vrms, surf_bright

    def call_jampy(
        self,
        surf_lum,
        sigma_lum,
        surf_mass,
        sigma_mass,
        x,
        y=None,
        q_lum=None,
        q_mass=None,
        inclination=90.0,
        beta=None,
        sigma_psf=0.0,
        norm_psf=1.0,
        pix_size=0.0,
        black_hole_mass=0.0,
        jam_kwargs=None,
    ):
        x = np.asarray(x)
        if y is None:
            y = np.zeros_like(x)
        y = np.asarray(y)
        x_shape = x.shape
        x = x.flatten()
        y = y.flatten()
        if (not self.axisymmetric) or (inclination is None):
            # spherical modeling
            r = np.sqrt(x**2 + y**2)
            vrms, surf_bright = self.call_jampy_sph(
                surf_lum,
                sigma_lum,
                surf_mass,
                sigma_mass,
                r,
                beta,
                sigma_psf,
                norm_psf,
                pix_size,
                black_hole_mass,
                jam_kwargs,
            )
        else:
            # axisymmetric modeling
            vrms, surf_bright = self.call_jampy_axi(
                surf_lum,
                sigma_lum,
                surf_mass,
                sigma_mass,
                x,
                y,
                q_lum,
                q_mass,
                inclination,
                beta,
                sigma_psf,
                norm_psf,
                pix_size,
                black_hole_mass,
                jam_kwargs,
            )
        vrms = vrms.reshape(x_shape)
        surf_bright = surf_bright.reshape(x_shape)
        return vrms, surf_bright

    def call_jampy_axi(
        self,
        surf_lum,
        sigma_lum,
        surf_mass,
        sigma_mass,
        x,
        y,
        q_lum=None,
        q_mass=None,
        inclination=90.0,
        beta=None,
        sigma_psf=0.0,
        norm_psf=1.0,
        pix_size=0.0,
        black_hole_mass=0.0,
        jam_kwargs=None,
    ):
        if jam_kwargs is None:
            jam_kwargs = {}
        jam_model = jam.axi.proj(
            surf_lum,
            sigma_lum,
            q_lum,
            surf_mass,
            sigma_mass,
            q_mass,
            xbin=x,
            ybin=y,
            inc=inclination,
            align=self.align,
            distance=self.cosmo.dd,
            beta=beta,
            logistic=self._anisotropy.use_logistic,
            mbh=black_hole_mass,
            sigmapsf=sigma_psf,
            normpsf=norm_psf,
            pixsize=pix_size,
            quiet=True,
            plot=False,
            **jam_kwargs,
        )
        vrms = jam_model.model
        surf_bright = jam_model.flux
        return vrms, surf_bright

    def call_jampy_sph(
        self,
        surf_lum,
        sigma_lum,
        surf_mass,
        sigma_mass,
        r,
        beta=None,
        sigma_psf=0.0,
        norm_psf=1.0,
        pix_size=0.0,
        black_hole_mass=0.0,
        jam_kwargs=None,
    ):
        if jam_kwargs is None:
            jam_kwargs = {}
        jam_model = jam.sph.proj(
            surf_lum,
            sigma_lum,
            surf_mass,
            sigma_mass,
            rad=r,
            distance=self.cosmo.dd,
            beta=beta,
            logistic=self._anisotropy.use_logistic,
            mbh=black_hole_mass,
            sigmapsf=sigma_psf,
            normpsf=norm_psf,
            pixsize=pix_size,
            quiet=True,
            plot=False,
            **jam_kwargs,
        )
        vrms = jam_model.model
        surf_bright = jam_model.flux
        return vrms, surf_bright

    @staticmethod
    def _extract_center(kwargs):
        # assumes that the center parameters are the same for all components
        if "center_x" in kwargs[0]:
            return kwargs[0]["center_x"], kwargs[0]["center_y"]
        else:
            return 0, 0

    @staticmethod
    def _extract_ellipticity(kwargs):
        # assumes that the ellipticity parameters are the same for all components
        if "e1" in kwargs[0]:
            return kwargs[0]["e1"], kwargs[0]["e2"]
        else:
            return 0, 0

    @staticmethod
    def _rotate_grid(x_grid, y_grid, phi):
        """Rotate the grid according to the ellipticity parameters.

        :param x_grid: x grid
        :param y_grid: y grid
        :param phi: angle in radians
        :return: x_rotated, y_rotated
        """
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        x_rotated = cos_phi * x_grid + sin_phi * y_grid
        y_rotated = -sin_phi * x_grid + cos_phi * y_grid

        return x_rotated, y_rotated

    def _shift_and_rotate(self, x, y, kwargs):
        center_x, center_y = self._extract_center(kwargs)
        x_shifted = x - center_x
        y_shifted = y - center_y
        e1, e2 = self._extract_ellipticity(kwargs)
        phi, q = ellipticity2phi_q(e1, e2)
        return self._rotate_grid(x_shifted, y_shifted, phi)
