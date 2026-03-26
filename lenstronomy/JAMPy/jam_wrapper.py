__author__ = "furcelay", "sibirrer"

import warnings

from lenstronomy.JAMPy.jam_wrapper_base import JAMWrapperBase
from lenstronomy.Util.param_util import ellipticity2phi_q
from lenstronomy.GalKin.observation import GalkinObservation
from astropy.stats import gaussian_fwhm_to_sigma
import numpy as np

__all__ = ["JAMWrapper"]


class JAMWrapper(JAMWrapperBase, GalkinObservation):
    """Wrapper class to use jampy JAM functionality similar to lenstronomy's Galkin
    class.

    :param kwargs_model: keyword arguments describing the model components
    :param kwargs_aperture: keyword arguments describing the spectroscopic aperture, see
        Aperture() class
    :param kwargs_psf: keyword argument specifying the PSF of the observation
    :param kwargs_cosmo: keyword arguments that define the cosmology in terms of the
        angular diameter distances involved
    """

    def __init__(
        self,
        kwargs_model,
        kwargs_aperture,
        kwargs_psf,
        kwargs_cosmo,
    ):

        JAMWrapperBase.__init__(self, kwargs_model, kwargs_cosmo)

        GalkinObservation.__init__(self, kwargs_aperture, kwargs_psf)

        if ("delta_pix" not in kwargs_aperture) and (
            "IFU" not in kwargs_aperture["aperture_type"]
        ):
            # set the sampling of the aperture to FWHM / 3
            kwargs_aperture = kwargs_aperture.copy()
            kwargs_aperture["delta_pix"] = min(self.psf_fwhm / 3, 0.1)

        if kwargs_aperture["aperture_type"] == "IFU_grid":
            kwargs_aperture = kwargs_aperture.copy()
            if "supersampling_factor" in kwargs_aperture:
                if (
                    kwargs_aperture["supersampling_factor"]
                    != self.psf_supersampling_factor
                ):
                    warnings.warn(
                        f"Supersampling factor in kwargs_aperture ({kwargs_aperture['supersampling_factor']}) "
                        f"does not match the one set by the PSF kwargs ({self.psf_supersampling_factor}). "
                        f"The one from the PSF kwargs will be used.",
                        UserWarning,
                    )
            kwargs_aperture["supersampling_factor"] = self.psf_supersampling_factor
            if "padding_arcsec" not in kwargs_aperture:
                # add a padding of 3 times the PSF sigma for convolution
                kwargs_aperture["padding_arcsec"] = (
                    gaussian_fwhm_to_sigma * self.psf_fwhm * 3
                )
            Aperture.__init__(self, **kwargs_aperture)
            self.convolution_padding = self._aperture.padding
        else:
            Aperture.__init__(self, **kwargs_aperture)
            self.convolution_padding = 0

    def dispersion(
        self,
        kwargs_mass,
        kwargs_light,
        kwargs_anisotropy,
        q_intrinsic=1.0,
        black_hole_mass=0.0,
        convolved=True,
        voronoi_bins=None,
    ):
        """Computes the velocity dispersion in the aperture. IF the aperture is a slit,
        frame or shell, the output is a single float. If the aperture is an IFU grid,
        the output is a 2D array of the same shape as the IFU grid. If the aperture is
        an IFU shells, the output is a 1D array with the number of shells.

        :param kwargs_mass: keyword arguments of the mass model
        :param kwargs_light: keyword argument of the light model
        :param kwargs_anisotropy: anisotropy keyword arguments
        :param q_intrinsic: intrinsic axis ratio of the light profile to compute the
            inclination angle
        :param black_hole_mass: mass of the central SMBH [solar masses]
        :param convolved: bool, if True the PSF convolution is applied
        :param voronoi_bins: None or 2D array with same shape as the IFU grid defining
            the Voronoi bins. If None, no Voronoi binning is applied. Only relevant if
            aperture is of type 'IFU_grid'.
        :return: ordered array of velocity dispersions [km/s] for each unit
        """
        x_sup, y_sup = self.aperture_sample()
        # shift and rotate to align with light profile
        x_gal_sup, y_gal_sup = self._shift_and_rotate(x_sup, y_sup, kwargs_light)
        inclination = self._get_inclination_angle(kwargs_light[0], q_intrinsic)
        if self.psf_type == "PIXEL":
            vrms_sup, surf_bright_sup = self.dispersion_points(
                x_gal_sup,
                y_gal_sup,
                kwargs_mass,
                kwargs_light,
                kwargs_anisotropy,
                inclination=inclination,
                black_hole_mass=black_hole_mass,
                convolved=False,
            )
            sigma2_lum_weighted_sup = vrms_sup**2 * surf_bright_sup
            if convolved:
                sigma2_lum_weighted_sup = self.convolve(sigma2_lum_weighted_sup)
                surf_bright_sup = self.convolve(surf_bright_sup)
        else:
            vrms_sup, surf_bright_sup = self.dispersion_points(
                x_gal_sup,
                y_gal_sup,
                kwargs_mass,
                kwargs_light,
                kwargs_anisotropy,
                inclination=inclination,
                convolved=convolved,
                black_hole_mass=black_hole_mass,
                psf_sigmas=self.psf_sigmas,
                psf_amplitudes=self.psf_amplitudes,
                delta_pix=self.delta_pix,
            )
            sigma2_lum_weighted_sup = vrms_sup**2 * surf_bright_sup

        if voronoi_bins is not None:
            if self.aperture_type != "IFU_grid":
                raise ValueError(
                    "Voronoi binning is only applicable for IFU_grid aperture type."
                )
            # this would be deprecated and replaced by IFU_binned aperture
            warnings.warn(
                "The voronoi bins keyword argument will be deprecated, "
                "use the IFU_binned aperture type instead.",
                DeprecationWarning,
            )
            sigma2_lum_weighted = downsample_cords_to_bins(
                sigma2_lum_weighted_sup,
                voronoi_bins,
                supersampling_factor=self.psf_supersampling_factor,
                padding=self.convolution_padding,
            )
            surf_bright = downsample_cords_to_bins(
                surf_bright_sup,
                voronoi_bins,
                supersampling_factor=self.psf_supersampling_factor,
                padding=self.convolution_padding,
            )
            vrms = np.sqrt(sigma2_lum_weighted / surf_bright)
        else:
            sigma2_lum_weighted = self.aperture_downsample(
                sigma2_lum_weighted_sup,
            )
            surf_bright = self.aperture_downsample(
                surf_bright_sup,
            )
            vrms = np.sqrt(sigma2_lum_weighted / surf_bright)
        return vrms

    def _get_inclination_angle(self, obs_kwargs, q_intrinsic):
        """Compute inclination angle from observed ellipticity and intrinsic axis ratio.

        :param obs_kwargs: dictionary with observed ellipticity parameters 'e1' and 'e2'
        :param q_intrinsic: intrinsic axis ratio
        :return: inclination angle in degrees
        """
        if (not self.axisymmetric) or q_intrinsic == 1.0:
            return 90.0  # spherical case
        e1_obs = obs_kwargs.get("e1", 0.0)
        e2_obs = obs_kwargs.get("e2", 0.0)
        phi_obs, q_obs = ellipticity2phi_q(e1_obs, e2_obs)
        if q_obs == 1.0:
            warnings.warn(
                "Cannot determine inclination angle for circular observed profile (q_obs=1.0)."
                " Spherical symmetry will be assumed.",
                UserWarning,
            )
            return None
        cos_i_squared = (q_obs**2 - q_intrinsic**2) / (1 - q_intrinsic**2)
        cos_i_squared = np.clip(cos_i_squared, 0, 1)
        inclination_angle = np.arccos(np.sqrt(cos_i_squared))
        return np.rad2deg(inclination_angle)
