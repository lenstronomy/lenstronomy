__author__ = 'sibirrer'

from lenstronomy.LensModel.single_plane import SinglePlane
from lenstronomy.LensModel.multi_plane import MultiPlane
from lenstronomy.Cosmo.lens_cosmo import LensCosmo


class LensModel(object):
    """
    class to handle an arbitrary list of lens models
    """

    def __init__(self, lens_model_list, z_lens=None, z_source=None, lens_redshift_list=None, cosmo=None,
                 multi_plane=False, numerical_alpha_class=None, observed_convention_index=None):
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
        """
        self.lens_model_list = lens_model_list
        self.z_lens = z_lens
        self.z_source = z_source
        self.redshift_list = lens_redshift_list
        self.cosmo = cosmo
        self.multi_plane = multi_plane
        if multi_plane is True:
            if z_source is None:
                raise ValueError('z_source needs to be set for multi-plane lens modelling.')
            self.lens_model = MultiPlane(z_source, lens_model_list, lens_redshift_list, cosmo=cosmo,
                                         numerical_alpha_class=numerical_alpha_class,
                                         observed_convention_index=observed_convention_index)
        else:
            self.lens_model = SinglePlane(lens_model_list, numerical_alpha_class=numerical_alpha_class)
        if z_lens is not None and z_source is not None:
            self._lensCosmo = LensCosmo(z_lens, z_source, cosmo=self.cosmo)

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
        :return:
        """
        if hasattr(self.lens_model, 'arrival_time'):
            arrival_time = self.lens_model.arrival_time(x_image, y_image, kwargs_lens)
        else:
            x_source, y_source = self.lens_model.ray_shooting(x_image, y_image, kwargs_lens)
            fermat_pot = self.lens_model.fermat_potential(x_image, y_image, x_source, y_source, kwargs_lens)
            if not hasattr(self, '_lensCosmo'):
                raise ValueError("LensModel class was not initalized with lens and source redshifts!")
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

    def alpha(self, x, y, kwargs, k=None):
        """
        deflection angles

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: deflection angles in units of arcsec
        """
        return self.lens_model.alpha(x, y, kwargs, k=k)

    def hessian(self, x, y, kwargs, k=None):
        """
        hessian matrix

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: f_xx, f_xy, f_yy components
        """
        return self.lens_model.hessian(x, y, kwargs, k=k)

    def kappa(self, x, y, kwargs, k=None):
        """
        lensing convergence k = 1/2 laplacian(phi)

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: lensing convergence
        """

        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs, k=k)
        kappa = 1./2 * (f_xx + f_yy)
        return kappa

    def gamma(self, x, y, kwargs, k=None):
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
        :return: gamma1, gamma2
        """

        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs, k=k)
        gamma1 = 1./2 * (f_xx - f_yy)
        gamma2 = f_xy
        return gamma1, gamma2

    def magnification(self, x, y, kwargs, k=None):
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
        :return: magnification
        """

        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs, k=k)
        det_A = (1 - f_xx) * (1 - f_yy) - f_xy*f_yx
        return 1./det_A  # attention, if dividing by zero

    def flexion(self, x, y, kwargs, diff=0.000001):
        """
        third derivatives (flexion)

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param diff: numerical differential length of Hessian
        :return: f_xxx, f_xxy, f_xyy, f_yyy
        """
        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs)

        f_xx_dx, f_xy_dx, f_yx_dx, f_yy_dx = self.hessian(x + diff, y, kwargs)
        f_xx_dy, f_xy_dy, f_yx_dy, f_yy_dy = self.hessian(x, y + diff, kwargs)

        f_xxx = (f_xx_dx - f_xx) / diff
        f_xxy = (f_xx_dy - f_xx) / diff
        f_xyy = (f_xy_dy - f_xy) / diff
        f_yyy = (f_yy_dy - f_yy) / diff
        return f_xxx, f_xxy, f_xyy, f_yyy
