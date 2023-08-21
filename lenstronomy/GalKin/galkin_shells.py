import numpy as np
from lenstronomy.GalKin.galkin import Galkin
from lenstronomy.Util import util
from lenstronomy.Util import mask_util
from scipy import signal


class GalkinShells(Galkin):
    """Class to calculate velocity dispersion for radial shells in a fast way."""

    def __init__(
        self,
        kwargs_model,
        kwargs_aperture,
        kwargs_psf,
        kwargs_cosmo,
        kwargs_numerics=None,
        analytic_kinematics=False,
    ):
        """:param kwargs_model: keyword arguments describing the model components :param
        kwargs_aperture: keyword arguments describing the spectroscopic aperture, see
        Aperture() class :param kwargs_psf: keyword argument specifying the PSF of the
        observation :param kwargs_cosmo: keyword arguments that define the cosmology in
        terms of the angular diameter distances involved :param kwargs_numerics:
        numerics keyword arguments :param analytic_kinematics: bool, if True uses the
        analytic kinematic model."""
        Galkin.__init__(
            self,
            kwargs_model=kwargs_model,
            kwargs_aperture=kwargs_aperture,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics,
            analytic_kinematics=analytic_kinematics,
        )
        if not self.aperture_type == "IFU_shells":
            raise ValueError(
                'GalkinShells is not supported with aperture_type %s. Only support with "IFU_shells"'
                % self.aperture_type
            )
        self._r_bins = kwargs_aperture["r_bins"]
        r_max = np.max(self._r_bins)
        self._num_pix = 100
        # factor of 1.5 to allow outside flux to be convolved into the largest bin
        self._delta_pix = 1.5 * r_max * 2 / self._num_pix
        x_grid, y_grid = util.make_grid(numPix=self._num_pix, deltapix=self._delta_pix)
        self._r_grid = np.sqrt(x_grid**2 + y_grid**2)

    def dispersion_map(self, kwargs_mass, kwargs_light, kwargs_anisotropy, **kwargs):
        """:param kwargs_mass: mass model parameters (following lenstronomy lens model
        conventions) :param kwargs_light: deflector light parameters (following
        lenstronomy light model conventions) :param kwargs_anisotropy: anisotropy
        parameters, may vary according to anisotropy type chosen.

        We refer to the Anisotropy() class for details on the parameters.
        :return: array of velocity dispersion for each IFU shell [km/s]
        """
        I_R_sigma2, IR = self.numerics._I_R_sigma2_interp(
            self._r_grid, kwargs_mass, kwargs_light, kwargs_anisotropy
        )
        ir_map = util.array2image(IR)
        ir_sigma2_map = util.array2image(I_R_sigma2)
        kernel = self.convolution_kernel(
            delta_pix=self._delta_pix, num_pix=self._num_pix
        )
        I_R_sigma2_conv = signal.fftconvolve(ir_sigma2_map, kernel, mode="same")
        I_R_sigma2_conv = util.image2array(I_R_sigma2_conv)
        I_R_conv = signal.fftconvolve(ir_map, kernel, mode="same")
        I_R_conv = util.image2array(I_R_conv)

        # average over radial bins
        vel_disp_array = []
        r_min = self._r_bins[0]
        for r_max in self._r_bins[1:]:
            mask = mask_util.mask_shell(
                self._r_grid, 0, center_x=0, center_y=0, r_in=r_min, r_out=r_max
            )
            vel_disp = np.sum(I_R_sigma2_conv * mask) / np.sum(
                I_R_conv * mask
            )  # luminosity weighted average
            vel_disp_array.append(vel_disp)
            r_min = r_max
        self.numerics.delete_cache()
        return np.sqrt(np.array(vel_disp_array)) / 1000
