__author__ = 'sibirrer'

from lenstronomy.LensModel.single_plane import SinglePlane
from lenstronomy.LensModel.multi_plane import MultiLens


class LensModel(object):
    """
    class to handle an arbitrary list of lens models
    """

    def __init__(self, lens_model_list, z_source=None, redshift_list=None, cosmo=None, multi_plane=False):
        """

        :param lens_model_list: list of strings with lens model names
        :param foreground_shear: bool, when True, models a foreground non-linear shear distortion
        """
        self.lens_model_list = lens_model_list
        self.z_source = z_source
        self.redshift_list = redshift_list
        self.cosmo = cosmo
        self.multi_plane = multi_plane
        if multi_plane is True:
            self.lens_model = MultiLens(z_source, lens_model_list, redshift_list, cosmo=cosmo)
        else:
            self.lens_model = SinglePlane(lens_model_list)

    def param_name_list(self):
        """
        returns the list of all parameter names

        :return: list of list of strings (for each light model separately)
        """
        name_list = []
        for func in self.lens_model.func_list:
            name_list.append(func.param_names)
        return name_list

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
        if self.multi_plane:
            raise ValueError("Fermat potential is not defined in multi-plane lensing. Please use single plane lens models.")
        else:
            return self.lens_model.fermat_potential(x_image, y_image, x_source, y_source, kwargs_lens)

    def arrival_time(self, x_image, y_image, kwargs_lens):
        """

        :param x_image:
        :param y_image:
        :param kwargs_lens:
        :return:
        """
        if self.multi_plane:
            return self.lens_model.arrival_time(x_image, y_image, kwargs_lens)
        else:
            raise ValueError(
                "arrival_time routine not defined for single plane lensing. Please use Fermat potential instead")

    def mass(self, x, y, epsilon_crit, kwargs):
        """

        :param x: position
        :param y: position
        :param epsilon_crit: critical mass density of a lens
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :return: projected mass density in units of input epsilon_crit
        """
        kappa = self.kappa(x, y, kwargs)
        mass = epsilon_crit * kappa
        return mass

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
        kappa = 1./2 * (f_xx + f_yy)  # attention on units
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
        gamma1 = 1./2 * (f_xx - f_yy)  # attention on units
        gamma2 = f_xy  # attention on units
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

    def mass_3d(self, r, kwargs, bool_list=None):
        """
        computes the mass within a 3d sphere of radius r

        :param r: radius (in angular units)
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param bool_list: list of bools that are part of the output
        :return: mass (in angular units, modulo epsilon_crit)
        """
        if self.multi_plane is True:
            raise ValueError("mass_3d is not supported for multi-lane lensing. Please use single plane instead.")
        else:
            return self.lens_model.mass_3d(r, kwargs, bool_list=bool_list)

    def mass_2d(self, r, kwargs, bool_list=None):
        """
        computes the mass enclosed a projected (2d) radius r

        :param r: radius (in angular units)
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param bool_list: list of bools that are part of the output
        :return: projected mass (in angular units, modulo epsilon_crit)
        """
        if self.multi_plane is True:
            raise ValueError("mass_2d is not supported for multi-lane lensing. Please use single plane instead.")
        else:
            return self.lens_model.mass_2d(r, kwargs, bool_list=bool_list)

