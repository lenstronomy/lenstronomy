__author__ = 'sibirrer'

# this file contains a class to compute the Navaro-Frank-White function in mass/kappa space
from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.Cosmo.lens_cosmo import LensCosmo


class NFWMC(object):
    """
    this class contains functions parameterises the NFW profile with log10 M200 and the concentration rs/r200
    relation are: R_200 = c * Rs

    ATTENTION: the parameterization is cosmology and redshift dependent!
    The cosmology to connect mass and deflection relations is fixed to default H0=70km/s Omega_m=0.3 flat LCDM.
    It is recommended to keep a given cosmology definition in the lens modeling as the observable reduced deflection
    angles are sensitive in this parameterization. If you do not want to impose a mass-concentration relation, it is
    recommended to use the default NFW lensing profile parameterized in reduced deflection angles.

    """
    param_names = ['logM', 'concentration', 'center_x', 'center_y']
    lower_limit_default = {'logM': 0, 'concentration': 0.01, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'logM': 16, 'concentration': 1000, 'center_x': 100, 'center_y': 100}

    def __init__(self, z_lens, z_source, cosmo=None):
        """

        :param z_lens: redshift of lens
        :param z_source: redshift of source
        :param cosmo: astropy cosmology instance
        """
        self._nfw = NFW()
        if cosmo is None:
            from astropy.cosmology import FlatLambdaCDM
            cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        self._lens_cosmo = LensCosmo(z_lens, z_source, cosmo=cosmo)

    def _m_c2deflections(self, logM, concentration):
        """

        :param logM: log10 mass in M200 stellar masses
        :param concentration: halo concentration c = r_200 / r_s
        :return: Rs (in arc seconds), alpha_Rs (in arc seconds)
        """
        M = 10 ** logM
        Rs, alpha_Rs = self._lens_cosmo.nfw_physical2angle(M, concentration)
        return Rs, alpha_Rs

    def function(self, x, y, logM, concentration, center_x=0, center_y=0):
        """

        :param x: angular position
        :param y: angular position
        :param Rs: angular turn over point
        :param alpha_Rs: deflection at Rs
        :param center_x: center of halo
        :param center_y: center of halo
        :return:
        """
        Rs, alpha_Rs = self._m_c2deflections(logM, concentration)
        return self._nfw.function(x, y, alpha_Rs=alpha_Rs, Rs=Rs, center_x=center_x, center_y=center_y)

    def derivatives(self, x, y, logM, concentration, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function (integral of NFW)
        """
        Rs, alpha_Rs = self._m_c2deflections(logM, concentration)
        return self._nfw.derivatives(x, y, Rs, alpha_Rs, center_x, center_y)

    def hessian(self, x, y, logM, concentration, center_x=0, center_y=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        Rs, alpha_Rs = self._m_c2deflections(logM, concentration)
        return self._nfw.hessian(x, y, Rs, alpha_Rs, center_x, center_y)
