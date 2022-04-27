__author__ = 'dgilman'

from lenstronomy.LensModel.lens_model import LensModel
import numpy as np


class MultiplaneFast(object):

    """
    This class accelerates ray tracing computations in multi plane lensing for quadruple image lenses by only
    computing the deflection from objects in front of the main deflector at z_lens one time. The first ray tracing
    computation through the foreground is saved and re-used, but it will always have the same shape as the initial
    x_image, y_image arrays.

    """

    def __init__(self, x_image, y_image, z_lens, z_source, lens_model_list, redshift_list,
                 astropy_instance, param_class, foreground_rays,
                 tol_source=1e-5, numerical_alpha_class=None):

        """

        :param x_image: x_image to fit
        :param y_image: y_image to fit
        :param z_lens: lens redshift
        :param z_source: source redshift
        :param lens_model_list: list of lens models
        :param redshift_list: list of lens redshifts
        :param astropy_instance: instance of astropy to pass to lens model
        :param param_class: an instance of ParamClass (see documentation in QuadOptimmizer.param_manager)
        :param foreground_rays: (optional) pre-computed foreground rays from a previous iteration, if they are not specified
        they will be re-computed
        :param tol_source: source plane chi^2 sigma
        :param numerical_alpha_class: class for computing numerically tabulated deflection angles
        """

        self.lensModel = LensModel(lens_model_list, z_lens, z_source, redshift_list, astropy_instance,
                                   multi_plane=True, numerical_alpha_class=numerical_alpha_class)

        lensmodel_list_to_vary = lens_model_list[0:param_class.to_vary_index]
        redshift_list_to_vary = redshift_list[0:param_class.to_vary_index]
        lensmodel_list_fixed = lens_model_list[param_class.to_vary_index:]
        redshift_list_fixed = redshift_list[param_class.to_vary_index:]

        self.lens_model_to_vary = LensModel(lensmodel_list_to_vary, z_lens, z_source, redshift_list_to_vary,
                                       cosmo=astropy_instance, multi_plane=True,
                                       numerical_alpha_class=numerical_alpha_class)

        self.lens_model_fixed = LensModel(lensmodel_list_fixed, z_lens, z_source, redshift_list_fixed,
                                            cosmo=astropy_instance, multi_plane=True,
                                            numerical_alpha_class=numerical_alpha_class)

        self._z_lens = z_lens

        self._z_source = z_source

        self._x_image = x_image
        self._y_image = y_image
        self._param_class = param_class

        self._tol_source = tol_source

        self._foreground_rays = foreground_rays

    def chi_square(self, args_lens, *args, **kwargs):

        """

        :param args_lens: array of lens model parameters being optimized, computed from kwargs_lens in a specified
         param_class, see documentation in QuadOptimizer.param_manager
        :return: total chi^2 penalty (source chi^2 + param chi^2), where param chi^2 is computed by the specified
         param_class
        """
        source_plane_penlty = self.source_plane_chi_square(args_lens)

        param_penalty = self._param_class.param_chi_square_penalty(args_lens)

        return source_plane_penlty + param_penalty

    def logL(self, args_lens, *args, **kwargs):

        """

        :param args_lens: array of lens model parameters being optimized, computed from kwargs_lens in a specified
         param_class, see documentation in QuadOptimizer.param_manager
        :return: the log likelihood corresponding to the given chi^2
        """
        chi_square = self.chi_square(args_lens)

        return -0.5 * chi_square

    def source_plane_chi_square(self, args_lens, *args, **kwargs):

        """

        :param args_lens: array of lens model parameters being optimized, computed from kwargs_lens in a specified
         param_class, see documentation in QuadOptimizer.param_manager
        :return: chi2 penalty for the source position (all images must map to the same source coordinate)
        """

        betax, betay = self.ray_shooting_fast(args_lens)

        dx_source = ((betax[0] - betax[1]) ** 2 + (betax[0] - betax[2]) ** 2 + (
                betax[0] - betax[3]) ** 2 + (
                             betax[1] - betax[2]) ** 2 +
                     (betax[1] - betax[3]) ** 2 + (betax[2] - betax[3]) ** 2)
        dy_source = ((betay[0] - betay[1]) ** 2 + (betay[0] - betay[2]) ** 2 + (
                betay[0] - betay[3]) ** 2 + (
                             betay[1] - betay[2]) ** 2 +
                     (betay[1] - betay[3]) ** 2 + (betay[2] - betay[3]) ** 2)

        chi_square = 0.5 * (dx_source + dy_source) / self._tol_source ** 2

        return chi_square

    def ray_shooting_fast(self, args_lens):

        """
        Performs a ray tracing computation through observed coordinates on the sky (self._x_image, self._y_image)
        to the source plane, returning the final coordinates of each ray on the source plane

        :param args_lens: An array of parameters being optimized. The array is computed from a set of key word arguments
         by an instance of ParamClass (see documentation in QuadOptimizer.param_manager)
        :return: the xy coordinate of each ray traced back to the source plane
        """

        # these do not depend on kwargs_lens_array
        x, y, alpha_x, alpha_y = self._ray_shooting_fast_foreground()

        # convert array into new kwargs dictionary
        kw = self._param_class.args_to_kwargs(args_lens)
        index = self._param_class.to_vary_index
        kwargs_lens = kw[0:index]
        # evaluate main deflector deflection angles
        x, y, alpha_x, alpha_y = self.lens_model_to_vary.lens_model.ray_shooting_partial(
            x, y, alpha_x, alpha_y, self._z_lens, self._z_lens, kwargs_lens, include_z_start=True)

        # ray trace through background halos
        kwargs_lens = kw[index:]
        x, y, _, _ = self.lens_model_fixed.lens_model.ray_shooting_partial(
            x, y, alpha_x, alpha_y, self._z_lens, self._z_source, kwargs_lens, check_convention=False)

        beta_x, beta_y = self.lens_model_fixed.lens_model.co_moving2angle_source(x, y)

        return beta_x, beta_y

    def _ray_shooting_fast_foreground(self):

        """
        Does the ray tracing through the foreground halos only once
        """

        if self._foreground_rays is None:

            # These do not depend on the kwargs being optimized for
            kw = self._param_class.kwargs_lens
            index = self._param_class.to_vary_index
            kwargs_lens = kw[index:]

            x0, y0 = np.zeros_like(self._x_image), np.zeros_like(self._y_image)
            x, y, alpha_x, alpha_y = self.lens_model_fixed.lens_model.ray_shooting_partial(
                x0, y0, self._x_image, self._y_image, z_start=0.,
                                                         z_stop=self._z_lens, kwargs_lens=kwargs_lens)

            self._foreground_rays = (x, y, alpha_x, alpha_y)

        return self._foreground_rays[0], self._foreground_rays[1], self._foreground_rays[2], self._foreground_rays[3]
