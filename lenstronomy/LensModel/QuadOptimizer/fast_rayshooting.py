__author__ = 'dgilman'

from lenstronomy.LensModel.lens_model import LensModel
import numpy as np


class FastRayShooting(object):

    """
    This class accelerates ray tracing computations by only computing the light rays from everything with z < z_lens once
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
        :param foreground_rays: pre-computed foreground rays from a previous iteration
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

        self.foreground_rays = foreground_rays

        self._tol_source = tol_source

    def source_plane_penalty(self, args_lens, *args, **kwargs):

        betax, betay = self.ray_shooting_fast(args_lens)

        dx_source = ((betax[0] - betax[1]) ** 2 + (betax[0] - betax[2]) ** 2 + (
                betax[0] - betax[3]) ** 2 + (
                             betax[1] - betax[2]) ** 2 +
                     (betax[1] - betax[3]) ** 2 + (betax[2] - betax[3]) ** 2)
        dy_source = ((betay[0] - betay[1]) ** 2 + (betay[0] - betay[2]) ** 2 + (
                betay[0] - betay[3]) ** 2 + (
                             betay[1] - betay[2]) ** 2 +
                     (betay[1] - betay[3]) ** 2 + (betay[2] - betay[3]) ** 2)

        return 0.5 * (dx_source + dy_source) / self._tol_source ** 2

    def ray_shooting_fast(self, args_lens):

        """

        :param args_lens: an array of parameters being optimized
        :return: the position of each ray in the source plane
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

        if self.foreground_rays is None:

            # These do not depend on the kwargs being optimized for
            kw = self._param_class.kwargs_lens
            index = self._param_class.to_vary_index
            kwargs_lens = kw[index:]

            x0, y0 = np.zeros_like(self._x_image), np.zeros_like(self._y_image)
            x, y, alpha_x, alpha_y = self.lens_model_fixed.lens_model.ray_shooting_partial(
                x0, y0, self._x_image, self._y_image, z_start=0.,
                                                         z_stop=self._z_lens, kwargs_lens=kwargs_lens)

            self.foreground_rays = (x, y, alpha_x, alpha_y)

        return self.foreground_rays[0], self.foreground_rays[1], self.foreground_rays[2], self.foreground_rays[3]
