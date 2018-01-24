import numpy as np
from lenstronomy.Cosmo.background import Background
from lenstronomy.LensModel.lens_model import LensModel


class MultiLens(object):
    """
    Multi-plnae lensing class
    """

    def __init__(self, z_source, lens_model_list, redshift_list, cosmo=None):
        """

        :param cosmo: instance of astropy.cosmology
        :return: Background class with instance of astropy.cosmology
        """
        from astropy.cosmology import default_cosmology

        if cosmo is None:
            cosmo = default_cosmology.get()
        self._cosmo_bkg = Background(cosmo)
        self._z_source = z_source
        if not len(lens_model_list) == len(redshift_list):
            raise ValueError("The length of lens_model_list does not correspond to redshift_list")
        self._lens_model_list = lens_model_list
        self._redshift_list = redshift_list
        self._sorted_redshift_index = self._index_ordering(redshift_list)
        self._lens_model = LensModel(lens_model_list)

    def ray_shooting(self, theta_x, theta_y, kwargs_lens):
        """
        ray-tracing (backwards light cone)

        :param theta_x: angle in x-direction on the image
        :param theta_y: angle in y-direction on the image
        :param kwargs_lens:
        :return: angles in the source plane
        """
        x = np.zeros_like(theta_x)
        y = np.zeros_like(theta_y)
        z_before = 0
        alpha_x = theta_x
        alpha_y = theta_y
        for idex in self._sorted_redshift_index:
            z_lens = self._redshift_list[idex]
            delta_T = self._cosmo_bkg.T_xy(z_before, z_lens)
            x, y = self._ray_step(x, y, alpha_x, alpha_y, delta_T)
            alpha_x, alpha_y = self._add_deflection(x, y, alpha_x, alpha_y, kwargs_lens, idex, z_lens)
            z_before = z_lens
        delta_T = self._cosmo_bkg.T_xy(z_before, self._z_source)
        x, y = self._ray_step(x, y, alpha_x, alpha_y, delta_T)
        beta_x, beta_y = self._co_moving2angle(x, y, self._z_source)
        return beta_x, beta_y

    def alpha(self, theta_x, theta_y, kwargs_lens):
        """
        reduced deflection angle computation

        :param theta_x: angle in x-direction
        :param theta_y: angle in y-direction
        :param kwargs_lens: lens model kwargs
        :return:
        """
        beta_x, beta_y = self.ray_shooting(theta_x, theta_y, kwargs_lens)
        alpha_x = theta_x - beta_x
        alpha_y = theta_y - beta_y
        return alpha_x, alpha_y

    def hessian(self, theta_x, theta_y, kwargs_lens, diff=0.000001):
        """
        computes the hessian components f_xx, f_yy, f_xy from f_x and f_y with numerical differentiation

        :param theta_x: x-position (preferentially arcsec)
        :type theta_x: numpy array
        :param theta_y: y-position (preferentially arcsec)
        :type theta_y: numpy array
        :param kwargs_lens: list of keyword arguments of lens model parameters matching the lens model classes
        :param diff: numerical differential step (float)
        :return: f_xx, f_xy, f_yx, f_yy
        """

        alpha_ra, alpha_dec = self.alpha(theta_x, theta_y, kwargs_lens)

        alpha_ra_dx, alpha_dec_dx = self.alpha(theta_x + diff, theta_y, kwargs_lens)
        alpha_ra_dy, alpha_dec_dy = self.alpha(theta_x, theta_y + diff, kwargs_lens)

        dalpha_rara = (alpha_ra_dx - alpha_ra)/diff
        dalpha_radec = (alpha_ra_dy - alpha_ra)/diff
        dalpha_decra = (alpha_dec_dx - alpha_dec)/diff
        dalpha_decdec = (alpha_dec_dy - alpha_dec)/diff

        f_xx = dalpha_rara
        f_yy = dalpha_decdec
        f_xy = dalpha_radec
        f_yx = dalpha_decra
        return f_xx, f_xy, f_yy

    def kappa(self, theta_x, theta_y, kwargs_lens):
        """
        lensing convergence k = 1/2 laplacian(phi)

        :param theta_x: x-position (preferentially arcsec)
        :type theta_x: numpy array
        :param theta_y: y-position (preferentially arcsec)
        :type theta_y: numpy array
        :param kwargs_lens: list of keyword arguments of lens model parameters matching the lens model classes
        :return: lensing convergence
        """

        f_xx, f_xy, f_yy = self.hessian(theta_x, theta_y, kwargs_lens)
        kappa = 1./2 * (f_xx + f_yy)  # attention on units
        return kappa

    def gamma(self, theta_x, theta_y, kwargs_lens):
        """
        shear computation
        g1 = 1/2(d^2phi/dx^2 - d^2phi/dy^2)
        g2 = d^2phi/dxdy

        :param theta_x: x-position (preferentially arcsec)
        :type theta_x: numpy array
        :param theta_y: y-position (preferentially arcsec)
        :type theta_y: numpy array
        :param kwargs_lens: list of keyword arguments of lens model parameters matching the lens model classes
        :return: gamma1, gamma2
        """

        f_xx, f_xy, f_yy = self.hessian(theta_x, theta_y, kwargs_lens)
        gamma1 = 1./2 * (f_xx - f_yy)  # attention on units
        gamma2 = f_xy  # attention on units
        return gamma1, gamma2

    def magnification(self, theta_x, theta_y, kwargs_lens):
        """
        magnification
        mag = 1/det(A)
        A = 1 - d^2phi/d_ij

        :param theta_x: x-position (preferentially arcsec)
        :type theta_x: numpy array
        :param theta_y: y-position (preferentially arcsec)
        :type theta_y: numpy array
        :param kwargs_lens: list of keyword arguments of lens model parameters matching the lens model classes
        :return: magnification
        """

        f_xx, f_xy, f_yy = self.hessian(theta_x, theta_y, kwargs_lens)
        det_A = (1 - f_xx) * (1 - f_yy) - f_xy*f_xy
        return 1./det_A  # attention, if dividing by zero

    def _index_ordering(self, redshift_list):
        """

        :param redshift_list: list of redshifts
        :return: indexes in acending order to be evaluated (from z=0 to z=z_source)
        """
        redshift_list = np.array(redshift_list)
        sort_index = np.argsort(redshift_list[redshift_list < self._z_source])
        if len(sort_index) < 1:
            raise ValueError("There is no lens object between observer at z=0 and source at z=%s" % self._z_source)
        return sort_index

    def _reduced2physical_deflection(self, alpha_reduced, z_lens, z_source):
        """
        alpha_reduced = D_ds/Ds alpha_physical

        :param alpha_reduced: reduced deflection angle
        :param z_lens: lens redshift
        :param z_source: source redshift
        :return: physical deflection angle
        """
        factor = self._cosmo_bkg.D_xy(0, z_source) / self._cosmo_bkg.D_xy(z_lens, z_source)
        return alpha_reduced * factor

    def _co_moving2angle(self, x, y, z_lens):
        """
        transforms co-moving distances Mpc into angles on the sky (radian)

        :param x: co-moving distance
        :param y: co-moving distance
        :param z_lens: redshift of plane
        :return: angles on the sky
        """
        T_z = self._cosmo_bkg.T_xy(0, z_lens)
        theta_x = x / T_z
        theta_y = y / T_z
        return theta_x, theta_y

    def _ray_step(self, x, y, alpha_x, alpha_y, delta_T):
        """
        ray propagation with small angle approximation

        :param x: co-moving x-position
        :param y: co-moving y-position
        :param alpha_x: deflection angle in x-direction at (x, y)
        :param alpha_y: deflection angle in y-direction at (x, y)
        :param delta_T: transversal angular diameter distance to the next step
        :return:
        """
        x_ = x + alpha_x * delta_T
        y_ = y + alpha_y * delta_T
        return x_, y_

    def _add_deflection(self, x, y, alpha_x, alpha_y, kwargs_lens, idex, z_lens):
        """
        adds the pyhsical deflection angle of a single lens plane to the deflection field

        :param x: co-moving distance at the deflector plane
        :param y: co-moving distance at the deflector plane
        :param alpha_x: physical angle (radian) before the deflector plane
        :param alpha_y: physical angle (radian) before the deflector plane
        :param kwargs_lens: lens model parameter kwargs
        :param idex: index of the lens model to be added
        :param z_lens: redshift of the deflector plane
        :return: updated physical deflection after deflector plane (in a backwards ray-tracing perspective)
        """
        theta_x, theta_y = self._co_moving2angle(x, y, z_lens)
        alpha_x_red, alpha_y_red = self._lens_model.alpha(theta_x, theta_y, kwargs_lens, k=idex)
        alpha_x_phys = self._reduced2physical_deflection(alpha_x_red, z_lens, z_source=self._z_source)
        alpha_y_phys = self._reduced2physical_deflection(alpha_y_red, z_lens, z_source=self._z_source)
        alpha_x_new = alpha_x - alpha_x_phys
        alpha_y_new = alpha_y - alpha_y_phys
        return alpha_x_new, alpha_y_new
