import numpy as np
from lenstronomy.Util import util
from lenstronomy.Util import mask as mask_util
import lenstronomy.Util.multi_gauss_expansion as mge


class LensProfileAnalysis(object):
    """
    class with analysis routines to compute derived properties of the lens model
    """
    def __init__(self, lens_model):
        """

        :param lens_model: LensModel instance
        """
        self._lens_model = lens_model

    def effective_einstein_radius(self, kwargs_lens, k=None, spacing=1000, get_precision=False, verbose=True):
        """
        computes the radius with mean convergence=1

        :param kwargs_lens: list of lens model keyword arguments
        :param spacing: number of annular bins to compute the convergence (resolution of the Einstein radius estimate)
        :param get_precision: If `True`, return the precision of estimated Einstein radius
        :return: estimate of the Einstein radius
        """
        if 'center_x' in kwargs_lens[0]:
            center_x = kwargs_lens[0]['center_x']
            center_y = kwargs_lens[0]['center_y']
        elif self._lens_model.lens_model_list[0] in ['INTERPOL', 'INTERPOL_SCALED']:
            center_x, center_y = 0, 0
        else:
            center_x, center_y = 0, 0
        numPix = 200
        deltaPix = 0.05
        x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltaPix)
        x_grid += center_x
        y_grid += center_y
        kappa = self._lens_model.kappa(x_grid, y_grid, kwargs_lens, k=k)
        if self._lens_model.lens_model_list[0] in ['INTERPOL', 'INTERPOL_SCALED']:
            center_x = x_grid[kappa == np.max(kappa)]
            center_y = y_grid[kappa == np.max(kappa)]
        kappa = util.array2image(kappa)
        r_array = np.linspace(0.0001, numPix*deltaPix/2., spacing)
        for r in r_array:
            mask = np.array(1 - mask_util.mask_center_2d(center_x, center_y, r, x_grid, y_grid))
            sum_mask = np.sum(mask)
            if sum_mask > 0:
                kappa_mean = np.sum(kappa*mask)/np.sum(mask)
                if kappa_mean < 1:
                    if get_precision:
                        return r, r_array[1] - r_array[0]
                    else:
                        return r
        if verbose:
            print(kwargs_lens, "Warning, no Einstein radius computed!")
        return np.nan #r_array[-1]

    def external_lensing_effect(self, kwargs_lens, lens_model_internal_bool=None):
        """
        computes deflection, shear and convergence at (0,0) for those part of the lens model not included in the main deflector

        :param kwargs_lens:
        :return:
        """
        alpha0_x, alpha0_y = 0, 0
        kappa_ext = 0
        shear1, shear2 = 0, 0
        if lens_model_internal_bool is None:
            lens_model_internal_bool = [True] * len(kwargs_lens)
        for i, kwargs in enumerate(kwargs_lens):
            if not lens_model_internal_bool[i] is True:
                f_x, f_y = self._lens_model.alpha(0, 0, kwargs_lens, k=i)
                f_xx, f_xy, f_yx, f_yy = self._lens_model.hessian(0, 0, kwargs_lens, k=i)
                alpha0_x += f_x
                alpha0_y += f_y
                kappa_ext += (f_xx + f_yy)/2.
                shear1 += 1./2 * (f_xx - f_yy)
                shear2 += f_xy
        return alpha0_x, alpha0_y, kappa_ext, shear1, shear2

    def profile_slope(self, kwargs_lens_list, lens_model_internal_bool=None, num_points=10, verbose=True):
        """
        computes the logarithmic power-law slope of a profile

        :param kwargs_lens_list: lens model keyword argument list
        :param lens_model_internal_bool: bool list, indicate which part of the model to consider
        :param num_points: number of estimates around the Einstein radius
        :return: logarithmic power-law slope
        """
        theta_E = self.effective_einstein_radius(kwargs_lens_list, verbose=verbose)
        if np.isnan(theta_E):
            if verbose:
                print("Could not compute effective slope, because of Einstein radius")
            return np.nan
        x0 = kwargs_lens_list[0]['center_x']
        y0 = kwargs_lens_list[0]['center_y']
        x, y = util.points_on_circle(theta_E, num_points)
        dr = 0.01
        x_dr, y_dr = util.points_on_circle(theta_E + dr, num_points)
        if lens_model_internal_bool is None:
            lens_model_internal_bool = [True]*len(kwargs_lens_list)

        alpha_E_x_i, alpha_E_y_i = self._lens_model.alpha(x0 + x, y0 + y, kwargs_lens_list, k=lens_model_internal_bool)
        alpha_E_r = np.sqrt(alpha_E_x_i**2 + alpha_E_y_i**2)
        alpha_E_dr_x_i, alpha_E_dr_y_i = self._lens_model.alpha(x0 + x_dr, y0 + y_dr, kwargs_lens_list,
                                                               k=lens_model_internal_bool)
        alpha_E_dr = np.sqrt(alpha_E_dr_x_i ** 2 + alpha_E_dr_y_i ** 2)
        slope = np.mean(np.log(alpha_E_dr / alpha_E_r) / np.log((theta_E + dr) / theta_E))
        gamma = -slope + 2
        return gamma

    def radial_lens_profile(self, r_list, kwargs_lens, center_x=None, center_y=None, model_bool_list=None):
        """

        :param r_list: list of radii to compute the spherically averaged lens light profile
        :param center_x: center of the profile
        :param center_y: center of the profile
        :param kwargs_lens_light: lens light parameter keyword argument list
        :param model_bool_list: bool list or None, indicating which profiles to sum over
        :return: flux amplitudes at r_list radii spherically averaged
        """
        if center_x is not None and center_y is not None:
            pass
        elif 'center_x' in kwargs_lens[0]:
            center_x = kwargs_lens[0]['center_x']
            center_y = kwargs_lens[0]['center_y']
        else:
            center_x, center_y = 0, 0
        kappa_list = []
        for r in r_list:
            x, y = util.points_on_circle(r, num_points=20)
            f_r = self._kappa_selected(x + center_x, y + center_y, kwargs_lens=kwargs_lens, model_bool_list=model_bool_list)
            kappa_list.append(np.average(f_r))
        return kappa_list

    def _kappa_selected(self, x_grid, y_grid, kwargs_lens, model_bool_list=None):
        """
        evaluates only part of the light profiles

        :param x_grid:
        :param y_grid:
        :param kwargs_lens_light:
        :return:
        """
        if model_bool_list is None:
            model_bool_list = [True] * len(kwargs_lens)
        kappa = np.zeros_like(x_grid)
        for i, bool in enumerate(model_bool_list):
            if bool is True:
                kappa_i = self._lens_model.kappa(x_grid, y_grid, kwargs_lens, k=i)
                kappa += kappa_i
        return kappa

    def multi_gaussian_lens(self, kwargs_lens, model_bool_list=None, n_comp=20):
        """
        multi-gaussian lens model in convergence space

        :param kwargs_lens:
        :param n_comp:
        :return:
        """
        if 'center_x' in kwargs_lens[0]:
            center_x = kwargs_lens[0]['center_x']
            center_y = kwargs_lens[0]['center_y']
        else:
            raise ValueError('no keyword center_x defined!')
        theta_E = self.effective_einstein_radius(kwargs_lens)
        r_array = np.logspace(-4, 2, 200) * theta_E
        kappa_s = self.radial_lens_profile(r_array, kwargs_lens, center_x=center_x, center_y=center_y,
                                                model_bool_list=model_bool_list)
        amplitudes, sigmas, norm = mge.mge_1d(r_array, kappa_s, N=n_comp)
        return amplitudes, sigmas, center_x, center_y

    def mass_fraction_within_radius(self, kwargs_lens, center_x, center_y, theta_E, numPix=100):
        """
        computes the mean convergence of all the different lens model components within a spherical aperture

        :param kwargs_lens: lens model keyword argument list
        :param center_x: center of the aperture
        :param center_y: center of the aperture
        :param theta_E: radius of aperture
        :return: list of average convergences for all the model components
        """
        x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=2.*theta_E / numPix)
        x_grid += center_x
        y_grid += center_y
        mask = mask_util.mask_sphere(x_grid, y_grid, center_x, center_y, theta_E)
        kappa_list = []
        for i in range(len(kwargs_lens)):
            kappa = self._lens_model.kappa(x_grid, y_grid, kwargs_lens, k=i)
            kappa_mean = np.sum(kappa * mask) / np.sum(mask)
            kappa_list.append(kappa_mean)
        return kappa_list

    def lens_center(self, kwargs_lens, k=None, bool_list=None, numPix=200, deltaPix=0.01, center_x_init=0, center_y_init=0):
        """
        computes the maximal convergence position on a grid and returns its coordinate

        :param kwargs_lens: lens model keyword argument list
        :param bool_list: bool list (optional) to include certain models or not
        :return: center_x, center_y
        """
        x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltaPix)
        x_grid += center_x_init
        y_grid += center_y_init

        if bool_list is None:
            kappa = self._lens_model.kappa(x_grid, y_grid, kwargs_lens, k=k)
        else:
            kappa = np.zeros_like(x_grid)
            for k in range(len(kwargs_lens)):
                if bool_list[k] is True:
                    kappa += self._lens_model.kappa(x_grid, y_grid, kwargs_lens, k=k)
        center_x = x_grid[kappa == np.max(kappa)]
        center_y = y_grid[kappa == np.max(kappa)]
        return center_x, center_y
