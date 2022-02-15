from __future__ import print_function, division, absolute_import, unicode_literals
__author__ = 'sibirrer'
from lenstronomy.LensModel.single_plane import SinglePlane
from lenstronomy.LensModel.MultiPlane.multi_plane import MultiPlane
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Util import constants as const

__all__ = ['LensModel']


class LensModel(object):
    """
    class to handle an arbitrary list of lens models. This is the main lenstronomy LensModel API for all other modules.
    """

    def __init__(self, lens_model_list, z_lens=None, z_source=None, lens_redshift_list=None, cosmo=None,
                 multi_plane=False, numerical_alpha_class=None, observed_convention_index=None,
                 z_source_convention=None, cosmo_interp=False, z_interp_stop=None, num_z_interp=100,
                 kwargs_interp=None):
        """

        :param lens_model_list: list of strings with lens model names
        :param z_lens: redshift of the deflector (only considered when operating in single plane mode).
        Is only needed for specific functions that require a cosmology.
        :param z_source: redshift of the source: Needed in multi_plane option only,
        not required for the core functionalities in the single plane mode.
        :param lens_redshift_list: list of deflector redshift (corresponding to the lens model list),
        only applicable in multi_plane mode.
        :param cosmo: instance of the astropy cosmology class. If not specified, uses the default cosmology.
        :param multi_plane: bool, if True, uses multi-plane mode. Default is False.
        :param numerical_alpha_class: an instance of a custom class for use in NumericalAlpha() lens model
        (see documentation in Profiles/numerical_alpha)
        :param kwargs_interp: interpolation keyword arguments specifying the numerics.
         See description in the Interpolate() class. Only applicable for 'INTERPOL' and 'INTERPOL_SCALED' models.
        :param observed_convention_index: a list of indices, corresponding to the lens_model_list element with same
        index, where the 'center_x' and 'center_y' kwargs correspond to observed (lensed) positions, not physical
        positions. The code will compute the physical locations when performing computations
        :param z_source_convention: float, redshift of a source to define the reduced deflection angles of the lens
        models. If None, 'z_source' is used.
        :param cosmo_interp: boolean (only employed in multi-plane mode), interpolates astropy.cosmology distances for
        faster calls when accessing several lensing planes
        :param z_interp_stop: (only in multi-plane with cosmo_interp=True); maximum redshift for distance interpolation
        This number should be higher or equal the maximum of the source redshift and/or the z_source_convention
        :param num_z_interp: (only in multi-plane with cosmo_interp=True); number of redshift bins for interpolating
        distances
        """
        self.lens_model_list = lens_model_list
        self.z_lens = z_lens
        self.z_source = z_source
        self._z_source_convention = z_source_convention
        self.redshift_list = lens_redshift_list

        if cosmo is None:
            from astropy.cosmology import default_cosmology
            cosmo = default_cosmology.get()
        self.cosmo = cosmo
        self.multi_plane = multi_plane
        if multi_plane is True:
            if z_source is None:
                raise ValueError('z_source needs to be set for multi-plane lens modelling.')

            self.lens_model = MultiPlane(z_source, lens_model_list, lens_redshift_list, cosmo=cosmo,
                                         numerical_alpha_class=numerical_alpha_class,
                                         observed_convention_index=observed_convention_index,
                                         z_source_convention=z_source_convention, cosmo_interp=cosmo_interp,
                                         z_interp_stop=z_interp_stop, num_z_interp=num_z_interp,
                                         kwargs_interp=kwargs_interp)
        else:
            self.lens_model = SinglePlane(lens_model_list, numerical_alpha_class=numerical_alpha_class,
                                          lens_redshift_list=lens_redshift_list,
                                          z_source_convention=z_source_convention, kwargs_interp=kwargs_interp)
        if z_lens is not None and z_source is not None:
            self._lensCosmo = LensCosmo(z_lens, z_source, cosmo=cosmo)

    def ray_shooting(self, x, y, kwargs, k=None):
        """
        maps image to source position (inverse deflection)

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: source plane positions corresponding to (x, y) in the image plane
        """
        return self.lens_model.ray_shooting(x, y, kwargs, k=k)

    def fermat_potential(self, x_image, y_image, kwargs_lens, x_source=None, y_source=None):
        """
        Fermat potential (negative sign means earlier arrival time)
        for Multi-plane lensing, it computes the effective Fermat potential (derived from the arrival time and
        subtracted off the time-delay distance for the given cosmology). The units are given in arcsecond square.

        :param x_image: image position
        :param y_image: image position
        :param x_source: source position
        :param y_source: source position
        :param kwargs_lens: list of keyword arguments of lens model parameters matching the lens model classes
        :return: fermat potential in arcsec**2 without geometry term (second part of Eqn 1 in Suyu et al. 2013) as a list
        """
        if hasattr(self.lens_model, 'fermat_potential'):
            return self.lens_model.fermat_potential(x_image, y_image, kwargs_lens, x_source, y_source)
        elif hasattr(self.lens_model, 'arrival_time') and hasattr(self, '_lensCosmo'):
            dt = self.lens_model.arrival_time(x_image, y_image, kwargs_lens)
            fermat_pot_eff = dt * const.c / self._lensCosmo.ddt / const.Mpc * const.day_s / const.arcsec ** 2
            return fermat_pot_eff
        else:
            raise ValueError('In multi-plane lensing you need to provide a specific z_lens and z_source for which the '
                             'effective Fermat potential is evaluated')

    def arrival_time(self, x_image, y_image, kwargs_lens, kappa_ext=0, x_source=None, y_source=None):
        """
        Arrival time of images relative to a straight line without lensing.
        Negative values correspond to images arriving earlier, and positive signs correspond to images arriving later.

        :param x_image: image position
        :param y_image: image position
        :param kwargs_lens: lens model parameter keyword argument list
        :param kappa_ext: external convergence contribution not accounted in the lens model that leads to the same
         observables in position and relative fluxes but rescales the time delays
        :param x_source: source position (optional), otherwise computed with ray-tracing
        :param y_source: source position (optional), otherwise computed with ray-tracing
        :return: arrival time of image positions in units of days
        """
        if hasattr(self.lens_model, 'arrival_time'):
            arrival_time = self.lens_model.arrival_time(x_image, y_image, kwargs_lens)
        else:
            fermat_pot = self.lens_model.fermat_potential(x_image, y_image, kwargs_lens, x_source=x_source,
                                                          y_source=y_source)
            if not hasattr(self, '_lensCosmo'):
                raise ValueError("LensModel class was not initialized with lens and source redshifts!")
            arrival_time = self._lensCosmo.time_delay_units(fermat_pot)
        arrival_time *= (1 - kappa_ext)
        return arrival_time

    def potential(self, x, y, kwargs, k=None):
        """
        lensing potential

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: lensing potential in units of arcsec^2
        """
        return self.lens_model.potential(x, y, kwargs, k=k)

    def alpha(self, x, y, kwargs, k=None, diff=None):
        """
        deflection angles

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :param diff: None or float. If set, computes the deflection as a finite numerical differential of the lensing
         potential. This differential is only applicable in the single lensing plane where the form of the lensing
         potential is analytically known
        :return: deflection angles in units of arcsec
        """
        if diff is None:
            return self.lens_model.alpha(x, y, kwargs, k=k)
        elif self.multi_plane is False:
            return self._deflection_differential(x, y, kwargs, k=k, diff=diff)
        else:
            raise ValueError('numerical differentiation of lensing potential is not available in the multi-plane '
                             'setting as analytical form of lensing potential is not available.')

    def hessian(self, x, y, kwargs, k=None, diff=None, diff_method='square'):
        """
        hessian matrix

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :param diff: float, scale over which the finite numerical differential is computed. If None, then using the
         exact (if available) differentials.
        :param diff_method: string, 'square' or 'cross', indicating whether finite differentials are computed from a
         cross or a square of points around (x, y)
        :return: f_xx, f_xy, f_yx, f_yy components
        """
        if diff is None:
            return self.lens_model.hessian(x, y, kwargs, k=k)
        elif diff_method == 'square':
            return self._hessian_differential_square(x, y, kwargs, k=k, diff=diff)
        elif diff_method == 'cross':
            return self._hessian_differential_cross(x, y, kwargs, k=k, diff=diff)
        elif diff_method == 'average':
            fxx_, fxy_, fyx_, fyy_ = self._hessian_differential_square(x, y, kwargs, k=k, diff=diff)
            _fxx, _fxy, _fyx, _fyy = self._hessian_differential_cross(x, y, kwargs, k=k, diff=diff)
            fxx = 0.5 * (fxx_ + _fxx)
            fxy = 0.5 * (fxy_ + _fxy)
            fyx = 0.5 * (fyx_ + _fyx)
            fyy = 0.5 * (fyy_ + _fyy)
            return fxx, fxy, fyx, fyy
        else:
            raise ValueError('diff_method %s not supported. Chose among "square" or "cross".' % diff_method)

    def kappa(self, x, y, kwargs, k=None, diff=None, diff_method='square'):
        """
        lensing convergence k = 1/2 laplacian(phi)

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :param diff: float, scale over which the finite numerical differential is computed. If None, then using the
         exact (if available) differentials.
        :param diff_method: string, 'square' or 'cross', indicating whether finite differentials are computed from a
         cross or a square of points around (x, y)
        :return: lensing convergence
        """

        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs, k=k, diff=diff, diff_method=diff_method)
        kappa = 1./2 * (f_xx + f_yy)
        return kappa

    def curl(self, x, y, kwargs, k=None, diff=None, diff_method='square'):
        """
        curl computation F_xy - F_yx

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :param diff: float, scale over which the finite numerical differential is computed. If None, then using the
         exact (if available) differentials.
        :param diff_method: string, 'square' or 'cross', indicating whether finite differentials are computed from a
         cross or a square of points around (x, y)
        :return: curl at position (x, y)
        """
        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs, k=k, diff=diff, diff_method=diff_method)
        return f_xy - f_yx

    def gamma(self, x, y, kwargs, k=None, diff=None, diff_method='square'):
        """
        shear computation
        g1 = 1/2(d^2phi/dx^2 - d^2phi/dy^2)
        g2 = d^2phi/dxdy

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :param diff: float, scale over which the finite numerical differential is computed. If None, then using the
         exact (if available) differentials.
        :param diff_method: string, 'square' or 'cross', indicating whether finite differentials are computed from a
         cross or a square of points around (x, y)
        :return: gamma1, gamma2
        """

        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs, k=k, diff=diff, diff_method=diff_method)
        gamma1 = 1./2 * (f_xx - f_yy)
        gamma2 = f_xy
        return gamma1, gamma2

    def magnification(self, x, y, kwargs, k=None, diff=None, diff_method='square'):
        """
        magnification
        mag = 1/det(A)
        A = 1 - d^2phi/d_ij

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :param diff: float, scale over which the finite numerical differential is computed. If None, then using the
         exact (if available) differentials.
        :param diff_method: string, 'square' or 'cross', indicating whether finite differentials are computed from a
         cross or a square of points around (x, y)
        :return: magnification
        """

        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs, k=k, diff=diff, diff_method=diff_method)
        det_A = (1 - f_xx) * (1 - f_yy) - f_xy*f_yx
        return 1./det_A  # attention, if dividing by zero

    def flexion(self, x, y, kwargs, k=None, diff=0.000001, hessian_diff=True):
        """
        third derivatives (flexion)

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: int or None, if set, only evaluates the differential from one model component
        :param diff: numerical differential length of Flexion
        :param hessian_diff: boolean, if true also computes the numerical differential length of Hessian (optional)
        :return: f_xxx, f_xxy, f_xyy, f_yyy
        """
        if hessian_diff is not True:
            hessian_diff = None
        f_xx_dx, f_xy_dx, f_yx_dx, f_yy_dx = self.hessian(x + diff/2, y, kwargs, k=k, diff=hessian_diff)
        f_xx_dy, f_xy_dy, f_yx_dy, f_yy_dy = self.hessian(x, y + diff/2, kwargs, k=k, diff=hessian_diff)

        f_xx_dx_, f_xy_dx_, f_yx_dx_, f_yy_dx_ = self.hessian(x - diff/2, y, kwargs, k=k, diff=hessian_diff)
        f_xx_dy_, f_xy_dy_, f_yx_dy_, f_yy_dy_ = self.hessian(x, y - diff/2, kwargs, k=k, diff=hessian_diff)

        f_xxx = (f_xx_dx - f_xx_dx_) / diff
        f_xxy = (f_xx_dy - f_xx_dy_) / diff
        f_xyy = (f_xy_dy - f_xy_dy_) / diff
        f_yyy = (f_yy_dy - f_yy_dy_) / diff
        return f_xxx, f_xxy, f_xyy, f_yyy

    def set_static(self, kwargs):
        """
        set this instance to a static lens model. This can improve the speed in evaluating lensing quantities at
        different positions but must not be used with different lens model parameters!

        :param kwargs: lens model keyword argument list
        :return: kwargs_updated (in case of image position convention in multiplane lensing this is changed)
        """
        return self.lens_model.set_static(kwargs)

    def set_dynamic(self):
        """
        deletes cache for static setting and makes sure the observed convention in the position of lensing profiles in
        the multi-plane setting is enabled. Dynamic is the default setting of this class enabling an accurate computation
        of lensing quantities with different parameters in the lensing profiles.

        :return: None
        """
        self.lens_model.set_dynamic()

    def _deflection_differential(self, x, y, kwargs, k=None, diff=0.00001):
        """

        :param x: x-coordinate
        :param y: y-coordinate
        :param kwargs: keyword argument list
        :param k: int or None, if set, only evaluates the differential from one model component
        :param diff: finite differential length
        :return: f_x, f_y
        """
        phi_dx = self.lens_model.potential(x + diff/2, y, kwargs=kwargs, k=k)
        phi_dy = self.lens_model.potential(x, y + diff/2, kwargs=kwargs, k=k)
        phi_dx_ = self.lens_model.potential(x - diff/2, y, kwargs=kwargs, k=k)
        phi_dy_ = self.lens_model.potential(x, y - diff/2, kwargs=kwargs, k=k)
        f_x = (phi_dx - phi_dx_) / diff
        f_y = (phi_dy - phi_dy_) / diff
        return f_x, f_y

    def _hessian_differential_cross(self, x, y, kwargs, k=None, diff=0.00001):
        """
        computes the numerical differentials over a finite range for f_xx, f_yy, f_xy from f_x and f_y
        The differentials are computed along the cross centered at (x, y).

        :param x: x-coordinate
        :param y: y-coordinate
        :param kwargs: lens model keyword argument list
        :param k: int, list of bools or None, indicating a subset of lens models to be evaluated
        :param diff: float, scale of the finite differential (diff/2 in each direction used to compute the differential
        :return: f_xx, f_xy, f_yx, f_yy
        """
        alpha_ra_dx, alpha_dec_dx = self.alpha(x + diff/2, y, kwargs, k=k)
        alpha_ra_dy, alpha_dec_dy = self.alpha(x, y + diff/2, kwargs, k=k)

        alpha_ra_dx_, alpha_dec_dx_ = self.alpha(x - diff/2, y, kwargs, k=k)
        alpha_ra_dy_, alpha_dec_dy_ = self.alpha(x, y - diff/2, kwargs, k=k)

        dalpha_rara = (alpha_ra_dx - alpha_ra_dx_) / diff
        dalpha_radec = (alpha_ra_dy - alpha_ra_dy_) / diff
        dalpha_decra = (alpha_dec_dx - alpha_dec_dx_) / diff
        dalpha_decdec = (alpha_dec_dy - alpha_dec_dy_) / diff

        f_xx = dalpha_rara
        f_yy = dalpha_decdec
        f_xy = dalpha_radec
        f_yx = dalpha_decra
        return f_xx, f_xy, f_yx, f_yy

    def _hessian_differential_square(self, x, y, kwargs, k=None, diff=0.00001):
        """
        computes the numerical differentials over a finite range for f_xx, f_yy, f_xy from f_x and f_y
        The differentials are computed on the square around (x, y). This minimizes curl.

        :param x: x-coordinate
        :param y: y-coordinate
        :param kwargs: lens model keyword argument list
        :param k: int, list of booleans or None, indicating a subset of lens models to be evaluated
        :param diff: float, scale of the finite differential (diff/2 in each direction used to compute the differential
        :return: f_xx, f_xy, f_yx, f_yy
        """
        alpha_ra_pp, alpha_dec_pp = self.alpha(x + diff/2, y + diff/2, kwargs, k=k)
        alpha_ra_pn, alpha_dec_pn = self.alpha(x + diff/2, y - diff/2, kwargs, k=k)

        alpha_ra_np, alpha_dec_np = self.alpha(x - diff / 2, y + diff / 2, kwargs, k=k)
        alpha_ra_nn, alpha_dec_nn = self.alpha(x - diff / 2, y - diff / 2, kwargs, k=k)

        f_xx = (alpha_ra_pp - alpha_ra_np + alpha_ra_pn - alpha_ra_nn) / diff / 2
        f_xy = (alpha_ra_pp - alpha_ra_pn + alpha_ra_np - alpha_ra_nn) / diff / 2
        f_yx = (alpha_dec_pp - alpha_dec_np + alpha_dec_pn - alpha_dec_nn) / diff / 2
        f_yy = (alpha_dec_pp - alpha_dec_pn + alpha_dec_np - alpha_dec_nn) / diff / 2

        return f_xx, f_xy, f_yx, f_yy
