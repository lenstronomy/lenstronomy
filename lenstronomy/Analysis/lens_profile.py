import numpy as np
import scipy
from lenstronomy.Util import util
from lenstronomy.Util import mask_util as mask_util
import lenstronomy.Util.multi_gauss_expansion as mge
from lenstronomy.Util import analysis_util
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

__all__ = ["LensProfileAnalysis"]


class LensProfileAnalysis(object):
    """Class with analysis routines to compute derived properties of the lens model."""

    def __init__(self, lens_model):
        """

        :param lens_model: LensModel instance
        """
        self._lens_model = lens_model

    def effective_einstein_radius_grid(
        self,
        kwargs_lens,
        center_x=None,
        center_y=None,
        model_bool_list=None,
        grid_num=200,
        grid_spacing=0.05,
        get_precision=False,
        verbose=True,
    ):
        """Computes the radius with mean convergence=1 on a grid.

        :param kwargs_lens: list of lens model keyword arguments
        :param center_x: position of the center (if not set, is attempting to find it from the parameters kwargs_lens)
        :param center_y: position of the center (if not set, is attempting to find it from the parameters kwargs_lens)
        :param model_bool_list: list of booleans indicating the addition (=True) of a model component in computing the
         Einstein radius
        :param grid_num: integer, number of grid points to numerically evaluate the convergence and estimate the
         Einstein radius
        :param grid_spacing: spacing in angular units of the grid
        :param get_precision: If `True`, return the precision of estimated Einstein radius
        :param verbose: if True, indicates warning when Einstein radius can not be computed
        :type verbose: bool
        :return: estimate of the Einstein radius
        """
        center_x, center_y = analysis_util.profile_center(
            kwargs_lens, center_x, center_y
        )

        x_grid, y_grid = util.make_grid(numPix=grid_num, deltapix=grid_spacing)
        x_grid += center_x
        y_grid += center_y
        kappa = self._lens_model.kappa(x_grid, y_grid, kwargs_lens, k=model_bool_list)
        if self._lens_model.lens_model_list[0] in ["INTERPOL", "INTERPOL_SCALED"]:
            center_x = x_grid[kappa == np.max(kappa)][0]
            center_y = y_grid[kappa == np.max(kappa)][0]

        return einstein_radius_from_grid(
            kappa,
            x_grid,
            y_grid,
            grid_spacing,
            grid_num,
            center_x=center_x,
            center_y=center_y,
            get_precision=get_precision,
            verbose=verbose,
        )

    def effective_einstein_radius(
        self, kwargs_lens, r_min=1e-3, r_max=1e1, num_points=30
    ):
        """Numerical estimate of the Einstein radius with integral approximation of
        radial convergence profile.

        :param kwargs_lens: list of lens model keyword arguments
        :param r_min: minimum radius of the convergence integrand
        :param r_max: maximum radius of the convergence integrand (should be larger than
            Einstein radius)
        :param num_points: number of radial points in log spacing
        :return: estimate of the Einstein radius
        """
        r_array = np.logspace(np.log10(r_min), np.log10(r_max), num_points)

        # Define the integrand function for the 1D numerical integration: this is the surface mass density
        # kappa at a given radius r, multiplied by 2*pi*r to account for the circular geometry.
        kappa_r = self.radial_lens_profile(r_array, kwargs_lens, center_x=0, center_y=0)

        # here we make a finer grid interpolation in log-log space
        k_interp = scipy.interpolate.interp1d(np.log10(r_array), np.log10(kappa_r))
        r_array = np.logspace(np.log10(r_min), np.log10(r_max), num_points * 10)
        kappa_r = 10 ** k_interp(np.log10(r_array))

        # we perform the integral in logarithmic steps of the convergence
        kappa_r = np.array(kappa_r)
        kappa_r_ = (kappa_r[1:] + kappa_r[:-1]) / 2
        r_array_ = (r_array[1:] + r_array[:-1]) / 2
        dlog_r = (np.log10(r_array[2]) - np.log10(r_array[1])) * np.log(10)
        # add the mass within the innermost bin and assume it's constant
        kappa_innermost = kappa_r[0] * np.pi * r_array[0] ** 2

        # the first part is the logarithmic integrand, the second part the circle integrand
        kappa_slice = kappa_r_ * dlog_r * r_array_ * (2 * np.pi * r_array_)
        kappa_slice = np.append(kappa_innermost, kappa_slice)

        kappa_cdf = np.cumsum(kappa_slice)
        # calculate average convergence at radius
        kappa_average = kappa_cdf / (np.pi * r_array**2)

        # we interpolate as the inverse function and evaluate this function for average kappa = 1
        # (assumes monotonic decline in average convergence)
        inv_interp = scipy.interpolate.interp1d(
            np.log10(kappa_average), np.log10(r_array)
        )
        try:
            theta_e = 10 ** inv_interp(0)
        except:
            theta_e = np.nan
        return theta_e

    def local_lensing_effect(
        self, kwargs_lens, ra_pos=0, dec_pos=0, model_list_bool=None
    ):
        """Computes deflection, shear and convergence at (ra_pos,dec_pos) for those part
        of the lens model not included in the main deflector.

        :param kwargs_lens: lens model keyword argument list
        :param ra_pos: RA position where to compute the external effect
        :param dec_pos: DEC position where to compute the external effect
        :param model_list_bool: boolean list indicating which models effect to be added
            to the estimate
        :return: alpha_x, alpha_y, kappa, shear1, shear2
        """
        f_x, f_y = self._lens_model.alpha(
            ra_pos, dec_pos, kwargs_lens, k=model_list_bool
        )
        f_xx, f_xy, f_yx, f_yy = self._lens_model.hessian(
            ra_pos, dec_pos, kwargs_lens, k=model_list_bool
        )
        kappa = (f_xx + f_yy) / 2.0
        shear1 = 1.0 / 2 * (f_xx - f_yy)
        shear2 = f_xy
        return f_x, f_y, kappa, shear1, shear2

    def profile_slope(
        self,
        kwargs_lens,
        radius,
        center_x=None,
        center_y=None,
        model_list_bool=None,
        num_points=10,
    ):
        """Computes the logarithmic power-law slope of a profile. ATTENTION: this is not
        an observable!

        :param kwargs_lens: lens model keyword argument list
        :param radius: radius from the center where to compute the logarithmic slope
            (angular units
        :param center_x: center of profile from where to compute the slope
        :param center_y: center of profile from where to compute the slope
        :param model_list_bool: bool list, indicate which part of the model to consider
        :param num_points: number of estimates around the Einstein radius
        :return: logarithmic power-law slope
        """
        center_x, center_y = analysis_util.profile_center(
            kwargs_lens, center_x, center_y
        )
        x, y = util.points_on_circle(radius, num_points)
        dr = 0.01
        x_dr, y_dr = util.points_on_circle(radius + dr, num_points)

        alpha_E_x_i, alpha_E_y_i = self._lens_model.alpha(
            center_x + x, center_y + y, kwargs_lens, k=model_list_bool
        )
        alpha_E_r = np.sqrt(alpha_E_x_i**2 + alpha_E_y_i**2)
        alpha_E_dr_x_i, alpha_E_dr_y_i = self._lens_model.alpha(
            center_x + x_dr, center_y + y_dr, kwargs_lens, k=model_list_bool
        )
        alpha_E_dr = np.sqrt(alpha_E_dr_x_i**2 + alpha_E_dr_y_i**2)
        slope = np.mean(np.log(alpha_E_dr / alpha_E_r) / np.log((radius + dr) / radius))
        gamma = -slope + 2
        return gamma

    def mst_invariant_differential(
        self,
        kwargs_lens,
        radius,
        center_x=None,
        center_y=None,
        model_list_bool=None,
        num_points=10,
    ):
        """Average of the radial stretch differential in radial direction, divided by
        the radial stretch factor.

        .. math::
            \\xi = \\frac{\\partial \\lambda_{\\rm rad}}{\\partial r} \\frac{1}{\\lambda_{\\rm rad}}

        This quantity is invariant under the MST.
        The specific definition is provided by Birrer 2021. Equivalent (proportional) definitions are provided by e.g.
        Kochanek 2020, Sonnenfeld 2018.

        :param kwargs_lens: lens model keyword argument list
        :param radius: radius from the center where to compute the MST invariant differential
        :param center_x: center position
        :param center_y: center position
        :param model_list_bool: indicate which part of the model to consider
        :param num_points: number of estimates around the radius
        :return: xi
        """
        center_x, center_y = analysis_util.profile_center(
            kwargs_lens, center_x, center_y
        )
        x, y = util.points_on_circle(radius, num_points)
        ext = LensModelExtensions(lensModel=self._lens_model)
        (
            lambda_rad,
            lambda_tan,
            orientation_angle,
            dlambda_tan_dtan,
            dlambda_tan_drad,
            dlambda_rad_drad,
            dlambda_rad_dtan,
            dphi_tan_dtan,
            dphi_tan_drad,
            dphi_rad_drad,
            dphi_rad_dtan,
        ) = ext.radial_tangential_differentials(
            x, y, kwargs_lens, center_x=center_x, center_y=center_y
        )
        xi = np.mean(dlambda_rad_drad / lambda_rad)
        return xi

    def radial_lens_profile(
        self, r_list, kwargs_lens, center_x=None, center_y=None, model_bool_list=None
    ):
        """

        :param r_list: list of radii to compute the spherically averaged lens light profile
        :param center_x: center of the profile
        :param center_y: center of the profile
        :param kwargs_lens: lens parameter keyword argument list
        :param model_bool_list: bool list or None, indicating which profiles to sum over
        :return: flux amplitudes at r_list radii azimuthally averaged
        """
        center_x, center_y = analysis_util.profile_center(
            kwargs_lens, center_x, center_y
        )
        kappa_list = []
        for r in r_list:
            x, y = util.points_on_circle(r, num_points=20)
            f_r = self._lens_model.kappa(
                x + center_x, y + center_y, kwargs=kwargs_lens, k=model_bool_list
            )
            kappa_list.append(np.average(f_r))
        return kappa_list

    def multi_gaussian_lens(
        self, kwargs_lens, center_x=None, center_y=None, model_bool_list=None, n_comp=20
    ):
        """Multi-gaussian lens model in convergence space.

        :param kwargs_lens:
        :param n_comp:
        :return:
        """
        center_x, center_y = analysis_util.profile_center(
            kwargs_lens, center_x, center_y
        )
        theta_E = self.effective_einstein_radius_grid(kwargs_lens)
        r_array = np.logspace(-4, 2, 200) * theta_E
        kappa_s = self.radial_lens_profile(
            r_array,
            kwargs_lens,
            center_x=center_x,
            center_y=center_y,
            model_bool_list=model_bool_list,
        )
        amplitudes, sigmas, norm = mge.mge_1d(r_array, kappa_s, N=n_comp)
        return amplitudes, sigmas, center_x, center_y

    def mass_fraction_within_radius(
        self, kwargs_lens, center_x, center_y, theta_E, numPix=100
    ):
        """Computes the mean convergence of all the different lens model components
        within a spherical aperture.

        :param kwargs_lens: lens model keyword argument list
        :param center_x: center of the aperture
        :param center_y: center of the aperture
        :param theta_E: radius of aperture
        :return: list of average convergences for all the model components
        """
        x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=2.0 * theta_E / numPix)
        x_grid += center_x
        y_grid += center_y
        mask = mask_util.mask_azimuthal(x_grid, y_grid, center_x, center_y, theta_E)
        kappa_list = []
        for i in range(len(kwargs_lens)):
            kappa = self._lens_model.kappa(x_grid, y_grid, kwargs_lens, k=i)
            kappa_mean = np.sum(kappa * mask) / np.sum(mask)
            kappa_list.append(kappa_mean)
        return kappa_list

    def convergence_peak(
        self,
        kwargs_lens,
        model_bool_list=None,
        grid_num=200,
        grid_spacing=0.01,
        center_x_init=0,
        center_y_init=0,
    ):
        """Computes the maximal convergence position on a grid and returns its
        coordinate.

        :param kwargs_lens: lens model keyword argument list
        :param model_bool_list: bool list (optional) to include certain models or not
        :return: center_x, center_y
        """
        x_grid, y_grid = util.make_grid(numPix=grid_num, deltapix=grid_spacing)
        x_grid += center_x_init
        y_grid += center_y_init

        kappa = self._lens_model.kappa(x_grid, y_grid, kwargs_lens, k=model_bool_list)

        center_x = x_grid[kappa == np.max(kappa)]
        center_y = y_grid[kappa == np.max(kappa)]
        return center_x, center_y


def einstein_radius_from_grid(
    kappa,
    x_grid,
    y_grid,
    grid_spacing,
    grid_num,
    center_x=0,
    center_y=0,
    get_precision=False,
    verbose=True,
):
    """Computes the radius with mean convergence=1.

    :param kappa: convergence calculated on a grid
    :param x_grid: x-value of grid points
    :param y_grid: y-value of grid points
    :param grid_spacing: spacing of grid points
    :param grid_num: number of grid points
    :param center_x: x-center of profile from where to measure circular averaged
        convergence
    :param center_y: y-center of profile from where to measure circular averaged
        convergence
    :param get_precision: if True, returns Einstein radius and expected numerical
        precision
    :param verbose: if True, indicates warning when Einstein radius can not be computed
    :type verbose: bool
    :return: einstein radius
    """

    r_array = np.linspace(start=0, stop=grid_num * grid_spacing / 2.0, num=grid_num * 2)
    inner_most_bin = True
    for r in r_array:
        mask = np.array(
            1 - mask_util.mask_center_2d(center_x, center_y, r, x_grid, y_grid)
        )
        sum_mask = np.sum(mask)
        if sum_mask > 0:
            kappa_mean = np.sum(kappa * mask) / np.sum(mask)
            if inner_most_bin:
                if kappa_mean < 1:
                    Warning(
                        "Central convergence value is subcritical <1 and hence an Einstein radius is ill defined."
                    )
                    if get_precision:
                        return np.nan, 0
                    else:
                        return np.nan
                inner_most_bin = False
            if kappa_mean < 1:
                if get_precision:
                    return r, r_array[1] - r_array[0]
                else:
                    return r
    if verbose:
        Warning(
            "Einstein radius could not be computed (or does not exist) for lens model."
        )
    if get_precision:
        return np.nan, 0
    else:
        return np.nan
