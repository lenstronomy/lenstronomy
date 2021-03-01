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

    def __init__(self, x_image, y_image, lensModel, lens_model_to_vary, lens_model_fixed,
                 param_class, foreground_rays=None, tol_source=1e-5):

        """
        This creates the class from a specified set of lens models that have already been created, thus saving
        memory if several instances of MultiplaneFast need to be created for a fixed lens model.

        :param x_image: x_image to fit
        :param y_image: y_image to fit
        :param lensModel: an instance of LensModel that contains every deflector in the lens system
        :param lens_model_to_vary: an instance of LensModel that contains only the lens models whose keywords are
        being sampled in the optimization
        :param lens_model_fixed: an instance of LensModel that contains the lens models whose properties are being
        held fixed during the optimization
        :param param_class: an instance of ParamClass (see documentation in QuadOptimmizer.param_manager)
        :param foreground_rays: (optional) pre-computed foreground rays from a previous iteration, if they are not specified
        they will be re-computed
        :param tol_source: source plane chi^2 sigma
        :param numerical_alpha_class: class for computing numerically tabulated deflection angles
        """

        self.lens_model_to_vary = lens_model_to_vary
        self.lensModel = lensModel
        self.lens_model_fixed = lens_model_fixed

        self._z_lens = lensModel.z_lens
        self._z_source = lensModel.z_source
        self._x_image = x_image
        self._y_image = y_image
        self._param_class = param_class
        self._tol_source = tol_source
        self._foreground_rays = foreground_rays

    @classmethod
    def fromModelList(cls, x_image, y_image, z_lens, z_source, lens_model_list, redshift_list,
                      astropy_instance, param_class, foreground_rays=None,
                      tol_source=1e-5, numerical_alpha_class=None):

        """
        This creates the class from a list of lens models and redshifts. The lens model list and redshift list
        will be split at the value of "to_vary_index" specified in the param_class (see classes in param_manager).
        Since this method creates several lens model classes it can consume significant memory.

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
        :return:
        """
        lensModel = LensModel(lens_model_list, z_lens, z_source, redshift_list, astropy_instance,
                              multi_plane=True, numerical_alpha_class=numerical_alpha_class)
        lensmodel_list_to_vary = lens_model_list[0:param_class.to_vary_index]
        redshift_list_to_vary = redshift_list[0:param_class.to_vary_index]
        lensmodel_list_fixed = lens_model_list[param_class.to_vary_index:]
        redshift_list_fixed = redshift_list[param_class.to_vary_index:]
        lens_model_to_vary = LensModel(lensmodel_list_to_vary, z_lens, z_source, redshift_list_to_vary,
                                       cosmo=astropy_instance, multi_plane=True,
                                       numerical_alpha_class=numerical_alpha_class)
        lens_model_fixed = LensModel(lensmodel_list_fixed, z_lens, z_source, redshift_list_fixed,
                                     cosmo=astropy_instance, multi_plane=True,
                                     numerical_alpha_class=numerical_alpha_class)

        return MultiplaneFast(x_image, y_image, lensModel, lens_model_to_vary, lens_model_fixed,
                              param_class, foreground_rays, tol_source)


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

    def alpha_fast(self, args_lens):
        """
        Performs a ray tracing computation through observed coordinates on the sky (self._x_image, self._y_image)
        to the source plane coordinate beta. Returns the deflection angle alpha:

        beta = x - alpha(x)

        :param args_lens: An array of parameters being optimized. The array is computed from a set of key word arguments
        by an instance of ParamClass (see documentation in QuadOptimizer.param_manager)
        :return: the xy coordinate of each ray traced back to the source plane
        """

        betax, betay = self.ray_shooting_fast(args_lens)
        alpha_x = self._x_image - betax
        alpha_y = self._y_image - betay

        return alpha_x, alpha_y

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

class MultiplaneFastDifferential(object):

    """
    This class uses the ray tracing routines in MultiPlaneFast to compute numerical derivatives of deflection angles
    i.e. the components of the hessian matrix
    """

    def __init__(self, diff, xcoords, ycoords, z_lens, z_source, lens_model_list, redshift_list,
                 astropy_instance, param_class, numerical_alpha_class=None):

        """

        :param diff: the angular scale over which to compute the finite difference derivative
        :param xcoords: the x coordinates where the derivatives are taken
        :param ycoords: the y coordinates where the derivatives are taken
        :param z_lens: the lens redshift
        :param z_source: the source redshift
        :param lens_model_list: the list of lens models to be passed to MultiPlaneFast
        :param redshift_list: the list of deflector redshifts to be passed to MuliPlaneFast
        :param astropy_instance: an instance as astropy
        :param param_class: the param class that defintes the function being minimized (see param_manager)
        :param numerical_alpha_class: (optional) a class that returns deflection angles for a numerically
        integrated mass profile
        """

        self._diff = diff

        self._fast_ray_shooting_dx_plus = MultiplaneFast.fromModelList(xcoords + diff / 2, ycoords, z_lens, z_source,
                                                                       lens_model_list, redshift_list, astropy_instance, param_class, None,
                                                                       numerical_alpha_class=numerical_alpha_class)

        lens_model_to_vary = self._fast_ray_shooting_dx_plus.lens_model_to_vary
        lens_model = self._fast_ray_shooting_dx_plus.lensModel
        lens_model_fixed = self._fast_ray_shooting_dx_plus.lens_model_fixed

        self._fast_ray_shooting_dy_plus = MultiplaneFast(xcoords, ycoords + diff / 2, lens_model, lens_model_to_vary,
                                                         lens_model_fixed, param_class)

        self._fast_ray_shooting_dx_minus = MultiplaneFast(xcoords - diff / 2, ycoords, lens_model, lens_model_to_vary,
                                                         lens_model_fixed, param_class)

        self._fast_ray_shooting_dy_minus = MultiplaneFast(xcoords, ycoords - diff / 2, lens_model, lens_model_to_vary,
                                                         lens_model_fixed, param_class)

        self._fast_ray_shooting_dx_plus_dy_plus = MultiplaneFast(xcoords + diff / 2, ycoords + diff / 2,
                                                                 lens_model, lens_model_to_vary,
                                                                 lens_model_fixed, param_class)

        self._fast_ray_shooting_dx_plus_dy_minus = MultiplaneFast(xcoords + diff/2, ycoords - diff/2, lens_model, lens_model_to_vary,
                                                         lens_model_fixed, param_class)

        self._fast_ray_shooting_dx_minus_dy_minus = MultiplaneFast(xcoords - diff / 2, ycoords - diff / 2, lens_model, lens_model_to_vary,
                                                         lens_model_fixed, param_class)

        self._fast_ray_shooting_dx_minus_dy_plus = MultiplaneFast(xcoords - diff / 2, ycoords + diff / 2, lens_model, lens_model_to_vary,
                                                         lens_model_fixed, param_class)

    def hessian(self, args, diff_method='square'):

        """

        :param args: the array of lens model args being optimized (see param_manager)
        :param diff_method: the method for calculating the derivatives, options include cross, square, and
        average, where average is the mean of cross and square
        :return: the derivatives of the deflection angles computed using the specified diff_methdd
        """

        if diff_method == 'cross':
            f_xx, f_xy, f_yx, f_yy = self._hessian_cross(args)
        elif diff_method == 'square':
            f_xx, f_xy, f_yx, f_yy = self._hessian_square(args)
        elif diff_method == 'average':
            _fxx, _fxy, _fyx, _fyy = self._hessian_cross(args)
            fxx_, fxy_, fyx_, fyy_ = self._hessian_square(args)
            f_xx = 0.5 * (fxx_ + _fxx)
            f_xy = 0.5 * (fxy_ + _fxy)
            f_yx = 0.5 * (fyx_ + _fyx)
            f_yy = 0.5 * (fyy_ + _fyy)
        else:
            raise Exception('diff method must be either cross, square, or average')

        return f_xx, f_xy, f_yx, f_yy

    def _hessian_cross(self, args):
        """

        :param args: the array of lens model args being optimized (see param_manager)
        :return: the derivatives of the deflection angles
        """

        alpha_ra_dx, alpha_dec_dx = self._fast_ray_shooting_dx_plus.alpha_fast(args)
        alpha_ra_dy, alpha_dec_dy = self._fast_ray_shooting_dy_plus.alpha_fast(args)

        alpha_ra_dx_, alpha_dec_dx_ = self._fast_ray_shooting_dx_minus.alpha_fast(args)
        alpha_ra_dy_, alpha_dec_dy_ = self._fast_ray_shooting_dy_minus.alpha_fast(args)

        dalpha_rara = (alpha_ra_dx - alpha_ra_dx_) / self._diff / 2
        dalpha_radec = (alpha_ra_dy - alpha_ra_dy_) / self._diff / 2
        dalpha_decra = (alpha_dec_dx - alpha_dec_dx_) / self._diff / 2
        dalpha_decdec = (alpha_dec_dy - alpha_dec_dy_) / self._diff / 2

        f_xx = dalpha_rara
        f_yy = dalpha_decdec
        f_xy = dalpha_radec
        f_yx = dalpha_decra

        return f_xx, f_xy, f_yx, f_yy

    def _hessian_square(self, args):
        """

        :param args: the array of lens model args being optimized (see param_manager)
        :return: the derivatives of the deflection angles
        """

        alpha_ra_pp, alpha_dec_pp = self._fast_ray_shooting_dx_plus_dy_plus.alpha_fast(args)
        alpha_ra_pn, alpha_dec_pn = self._fast_ray_shooting_dx_plus_dy_minus.alpha_fast(args)

        alpha_ra_np, alpha_dec_np = self._fast_ray_shooting_dx_minus_dy_plus.alpha_fast(args)
        alpha_ra_nn, alpha_dec_nn = self._fast_ray_shooting_dx_minus_dy_minus.alpha_fast(args)

        f_xx = (alpha_ra_pp - alpha_ra_np + alpha_ra_pn - alpha_ra_nn) / self._diff / 2
        f_xy = (alpha_ra_pp - alpha_ra_pn + alpha_ra_np - alpha_ra_nn) / self._diff / 2
        f_yx = (alpha_dec_pp - alpha_dec_np + alpha_dec_pn - alpha_dec_nn) / self._diff / 2
        f_yy = (alpha_dec_pp - alpha_dec_pn + alpha_dec_np - alpha_dec_nn) / self._diff / 2

        return f_xx, f_xy, f_yx, f_yy




