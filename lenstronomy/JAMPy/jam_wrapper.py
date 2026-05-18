__author__ = "furcelay", "sibirrer"

import warnings
from lenstronomy.JAMPy.jam_wrapper_base import JAMWrapperBase
from lenstronomy.GalKin.observation import GalkinObservation
from lenstronomy.GalKin.aperture import downsample_values_to_bins
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
        GalkinObservation.__init__(self, kwargs_aperture, kwargs_psf, backend="jampy")

    def dispersion(
        self,
        kwargs_mass,
        kwargs_light,
        kwargs_anisotropy,
        inclination=90.0,
        black_hole_mass=0.0,
        convolved=True,
        voronoi_bins=None,
        supersampling_factor=None,
    ):
        """Computes the velocity dispersion in the aperture. IF the aperture is a slit,
        frame or shell, the output is a single float. If the aperture is an IFU grid,
        the output is a 2D array of the same shape as the IFU grid. If the aperture is
        an IFU shells, the output is a 1D array with the number of shells.

        :param kwargs_mass: keyword arguments of the mass model
        :param kwargs_light: keyword argument of the light model
        :param kwargs_anisotropy: anisotropy keyword arguments
        :param inclination: inclination angle of the lens galaxy [degrees]
        :param black_hole_mass: mass of the central SMBH [solar masses]
        :param convolved: bool, if True the PSF convolution is applied
        :param voronoi_bins: None or 2D array with same shape as the IFU grid defining
            the Voronoi bins. If None, no Voronoi binning is applied. Only relevant if
            aperture is of type 'IFU_grid'.
        :param supersampling_factor: supersampling factor of the aperture. Only relevant
            for a PIXEL PSF, otherwise supersampling is done within jampy
        :return: ordered array of velocity dispersions [km/s] for each unit
        """
        if supersampling_factor is None:
            supersampling_factor = self._default_supersampling_factor
        x_sup, y_sup = self.aperture_sample(supersampling_factor)
        # shift and rotate to align with light profile
        x_gal_sup, y_gal_sup = self._shift_and_rotate(x_sup, y_sup, kwargs_light)
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
                sigma2_lum_weighted_sup = self.convolve(
                    sigma2_lum_weighted_sup, supersampling_factor
                )
                surf_bright_sup = self.convolve(surf_bright_sup, supersampling_factor)
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
                psf_sigmas=self.psf_multi_gauss_sigmas,
                psf_amplitudes=self.psf_multi_gauss_amplitudes,
                delta_pix=self.delta_pix,
            )
            sigma2_lum_weighted_sup = vrms_sup**2 * surf_bright_sup

        sigma2_lum_weighted = self.aperture_downsample(
            sigma2_lum_weighted_sup, supersampling_factor
        )
        surf_bright = self.aperture_downsample(surf_bright_sup, supersampling_factor)

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
            sigma2_lum_weighted = downsample_values_to_bins(
                sigma2_lum_weighted,
                voronoi_bins,
            )
            surf_bright = downsample_values_to_bins(
                surf_bright,
                voronoi_bins,
            )
        vrms = np.sqrt(sigma2_lum_weighted / surf_bright)
        return vrms
