"""This module contains a class to compute the elliptical Navarro-Frank-White function
in mass/kappa space."""

__author__ = "ajshajib"

from lenstronomy.LensModel.Profiles.nfw_mass_concentration import NFWMC
from lenstronomy.LensModel.Profiles.nfw_ellipse import NFW_ELLIPSE


class NFWMCEllipse(NFWMC):
    """This class contains functions parameterises the NFW profile with log10 M200 and
    the concentration rs/r200 relation are: R_200 = c * Rs.

    ATTENTION: the parameterization is cosmology and redshift dependent!
    The cosmology to connect mass and deflection relations is fixed to default H0=70km/s
    Omega_m=0.3 flat LCDM. It is recommended to keep a given cosmology definition in the
    lens modeling as the observable reduced deflection angles are sensitive in this
    parameterization. If you do not want to impose a mass-concentration relation, it is
    recommended to use the default NFW lensing profile parameterized in reduced
    deflection angles.
    """

    param_names = ["logM", "concentration", "e1", "e2", "center_x", "center_y"]
    lower_limit_default = {
        "logM": 0,
        "concentration": 0.01,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "logM": 16,
        "concentration": 1000,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(self, z_lens, z_source, cosmo=None, static=False):
        """

        :param z_lens: redshift of lens
        :param z_source: redshift of source
        :param cosmo: astropy cosmology instance
        :param static: boolean, if True, only operates with fixed parameter values
        """
        super(NFWMCEllipse, self).__init__(z_lens, z_source, cosmo, static)
        self._nfw = NFW_ELLIPSE()

    def function(self, x, y, logM, concentration, e1, e2, center_x=0, center_y=0):
        """Compute the lensing potential of the NFW profile with ellipticity.

        :param x: angular position
        :param y: angular position
        :param Rs: angular turn over point
        :param alpha_Rs: deflection at Rs
        :param e1: eccentricity component in x-direction
        :param e2: eccentricity component in y-direction
        :param center_x: center of halo
        :param center_y: center of halo
        :return:
        """
        Rs, alpha_Rs = self._m_c2deflections(logM, concentration)
        return self._nfw.function(
            x,
            y,
            alpha_Rs=alpha_Rs,
            Rs=Rs,
            e1=e1,
            e2=e2,
            center_x=center_x,
            center_y=center_y,
        )

    def derivatives(self, x, y, logM, concentration, e1, e2, center_x=0, center_y=0):
        """Return df/dx and df/dy of the function (integral of NFW).

        :param x: angular position
        :param y: angular position
        :param Rs: angular turn over point
        :param alpha_Rs: deflection at Rs
        :param e1: eccentricity component in x-direction
        :param e2: eccentricity component in y-direction
        :param center_x: center of halo
        :param center_y: center of halo
        :return:
        """
        Rs, alpha_Rs = self._m_c2deflections(logM, concentration)
        return self._nfw.derivatives(x, y, Rs, alpha_Rs, e1, e2, center_x, center_y)

    def hessian(self, x, y, logM, concentration, e1, e2, center_x=0, center_y=0):
        """Return Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx, d^f/dy^2.

        :param x: angular position
        :param y: angular position
        :param Rs: angular turn over point
        :param alpha_Rs: deflection at Rs
        :param e1: eccentricity component in x-direction
        :param e2: eccentricity component in y-direction
        :param center_x: center of halo
        :param center_y: center of halo
        :return:
        """
        Rs, alpha_Rs = self._m_c2deflections(logM, concentration)
        return self._nfw.hessian(x, y, Rs, alpha_Rs, e1, e2, center_x, center_y)
