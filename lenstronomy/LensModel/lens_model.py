from __future__ import print_function, division, absolute_import, unicode_literals
__author__ = 'sibirrer'
from lenstronomy.LensModel.single_plane import SinglePlane
from lenstronomy.LensModel.multi_plane import MultiPlane
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from astropy.cosmology import default_cosmology


class LensModel(object):
    """
    class to handle an arbitrary list of lens models
    """

    def __init__(self, lens_model_list, z_lens=None, z_source=None, lens_redshift_list=None, cosmo=None,
                 multi_plane=False, numerical_alpha_class=None, observed_convention_index=None, z_source_convention=None):
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
        :param observed_convention_index: a list of lens indexes that correspond to observed positions on the sky, not
        physical positions
        :param z_source_convention: float, redshift of a source to define the reduced deflection angles of the lens
        models. If None, 'z_source' is used.
        """
        self.lens_model_list = lens_model_list
        self.z_lens = z_lens
        if z_source_convention is None:
            z_source_convention = z_source
        self.z_source = z_source
        self._z_source_convention = z_source_convention
        self.redshift_list = lens_redshift_list

        if cosmo is None:
            cosmo = default_cosmology.get()
        self.cosmo = cosmo
        self.multi_plane = multi_plane
        if multi_plane is True:
            if z_source is None:
                raise ValueError('z_source needs to be set for multi-plane lens modelling.')

            self.lens_model = MultiPlane(z_source, lens_model_list, lens_redshift_list, cosmo=cosmo,
                                         numerical_alpha_class=numerical_alpha_class,
                                         observed_convention_index=observed_convention_index,
                                         z_source_convention=z_source_convention)
        else:
            self.lens_model = SinglePlane(lens_model_list, numerical_alpha_class=numerical_alpha_class,
                                          lens_redshift_list=lens_redshift_list, z_source_convention=z_source_convention)
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

    def fermat_potential(self, x_image, y_image, x_source, y_source, kwargs_lens):
        """
        fermat potential (negative sign means earlier arrival time)

        :param x_image: image position
        :param y_image: image position
        :param x_source: source position
        :param y_source: source position
        :param kwargs_lens: list of keyword arguments of lens model parameters matching the lens model classes
        :return: fermat potential in arcsec**2 without geometry term (second part of Eqn 1 in Suyu et al. 2013) as a list
        """
        if hasattr(self.lens_model, 'fermat_potential'):
            return self.lens_model.fermat_potential(x_image, y_image, x_source, y_source, kwargs_lens)
        else:
            raise ValueError("Fermat potential is not defined in multi-plane lensing. Please use single plane lens models.")

    def arrival_time(self, x_image, y_image, kwargs_lens):
        """

        :param x_image: image position
        :param y_image: image position
        :param kwargs_lens: lens model parameter keyword argument list
        :return: arrival time of image positions in units of days
        """
        try:
            arrival_time = self.lens_model.arrival_time(x_image, y_image, kwargs_lens)
        except:
            x_source, y_source = self.lens_model.ray_shooting(x_image, y_image, kwargs_lens)
            fermat_pot = self.lens_model.fermat_potential(x_image, y_image, x_source, y_source, kwargs_lens)
            if not hasattr(self, '_lensCosmo'):
                raise ValueError("LensModel class was not initialized with lens and source redshifts!")
            arrival_time = self._lensCosmo.time_delay_units(fermat_pot)
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

    def hessian(self, x, y, kwargs, k=None, diff=None):
        """
        hessian matrix

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :param diff: float, scale over which the finite numerical differential is computed. If None, then using the exact (if available) differentials.
        :return: f_xx, f_xy, f_yy components
        """
        if diff is None:
            return self.lens_model.hessian(x, y, kwargs, k=k)
        else:
            return self._hessian_differential(x, y, kwargs, k=k, diff=diff)

    def kappa(self, x, y, kwargs, k=None, diff=None):
        """
        lensing convergence k = 1/2 laplacian(phi)

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :param diff: float, scale over which the finite numerical differential is computed. If None, then using the exact (if available) differentials.
        :return: lensing convergence
        """

        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs, k=k, diff=diff)
        kappa = 1./2 * (f_xx + f_yy)
        return kappa

    def curl(self, x, y, kwargs, k=None, diff=None):
        """
        curl computation F_xy - F_yx

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :param diff: float, scale over which the finite numerical differential is computed. If None, then using the exact (if available) differentials.
        :return: curl at position (x, y)
        """
        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs, k=k, diff=diff)
        return f_xy - f_yx

    def gamma(self, x, y, kwargs, k=None, diff=None):
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
        :param diff: float, scale over which the finite numerical differential is computed. If None, then using the exact (if available) differentials.
        :return: gamma1, gamma2
        """

        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs, k=k, diff=diff)
        gamma1 = 1./2 * (f_xx - f_yy)
        gamma2 = f_xy
        return gamma1, gamma2

    def magnification(self, x, y, kwargs, k=None, diff=None):
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
        :param diff: float, scale over which the finite numerical differential is computed. If None, then using the exact (if available) differentials.
        :return: magnification
        """

        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs, k=k, diff=diff)
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
        :param diff: numerical differential length of Flexion
        :param hessian_diff: boolean, if true also computes the numerical differential length of Hessian (optional)
        :return: f_xxx, f_xxy, f_xyy, f_yyy
        """
        #f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs, k=k, diff=hessian_diff)
        if hessian_diff is not True:
            hessian_diff = None
        f_xx_dx, f_xy_dx, f_yx_dx, f_yy_dx = self.hessian(x + diff, y, kwargs, k=k, diff=hessian_diff)
        f_xx_dy, f_xy_dy, f_yx_dy, f_yy_dy = self.hessian(x, y + diff, kwargs, k=k, diff=hessian_diff)

        f_xx_dx_, f_xy_dx_, f_yx_dx_, f_yy_dx_ = self.hessian(x - diff, y, kwargs, k=k, diff=hessian_diff)
        f_xx_dy_, f_xy_dy_, f_yx_dy_, f_yy_dy_ = self.hessian(x, y - diff, kwargs, k=k, diff=hessian_diff)

        f_xxx = (f_xx_dx - f_xx_dx_) / diff / 2
        f_xxy = (f_xx_dy - f_xx_dy_) / diff / 2
        f_xyy = (f_xy_dy - f_xy_dy_) / diff / 2
        f_yyy = (f_yy_dy - f_yy_dy_) / diff / 2
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
        the multiplane setting is enabled. Dynamic is the default setting of this class enabling an accurate computation
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
        phi_dx = self.lens_model.potential(x + diff, y, kwargs=kwargs, k=k)
        phi_dy = self.lens_model.potential(x, y + diff, kwargs=kwargs, k=k)
        phi_dx_ = self.lens_model.potential(x - diff, y, kwargs=kwargs, k=k)
        phi_dy_ = self.lens_model.potential(x, y - diff, kwargs=kwargs, k=k)
        f_x = (phi_dx - phi_dx_) / diff / 2
        f_y = (phi_dy - phi_dy_) / diff / 2
        return f_x, f_y

    def _hessian_differential(self, x, y, kwargs, k=None, diff=0.00001):
        """
        computes the numerical differentials over a finite range for f_xx, f_yy, f_xy from f_x and f_y

        :param x: x-coordinate
        :param y: y-coordinate
        :return: f_xx, f_xy, f_yx, f_yy
        """
        #alpha_ra, alpha_dec = self.alpha(x, y, kwargs, k=k)

        alpha_ra_dx, alpha_dec_dx = self.alpha(x + diff, y, kwargs, k=k)
        alpha_ra_dy, alpha_dec_dy = self.alpha(x, y + diff, kwargs, k=k)

        alpha_ra_dx_, alpha_dec_dx_ = self.alpha(x - diff, y, kwargs, k=k)
        alpha_ra_dy_, alpha_dec_dy_ = self.alpha(x, y - diff, kwargs, k=k)

        dalpha_rara = (alpha_ra_dx - alpha_ra_dx_)/diff/2
        dalpha_radec = (alpha_ra_dy - alpha_ra_dy_)/diff/2
        dalpha_decra = (alpha_dec_dx - alpha_dec_dx_)/diff/2
        dalpha_decdec = (alpha_dec_dy - alpha_dec_dy_)/diff/2

        f_xx = dalpha_rara
        f_yy = dalpha_decdec
        f_xy = dalpha_radec
        f_yx = dalpha_decra
        return f_xx, f_xy, f_yx, f_yy
