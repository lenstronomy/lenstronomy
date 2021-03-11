__author__ = 'sibirrer'

# this file contains a class to compute the Navaro-Frank-White function in mass/kappa space
from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ['NFWMC']


class NFWMC(LensProfileBase):
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

    def __init__(self, z_lens, z_source, cosmo=None, static=False):
        """

        :param z_lens: redshift of lens
        :param z_source: redshift of source
        :param cosmo: astropy cosmology instance
        :param static: boolean, if True, only operates with fixed parameter values
        """
        self._nfw = NFW()
        if cosmo is None:
            from astropy.cosmology import FlatLambdaCDM
            cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        self._lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo)
        self._static = static
        super(NFWMC, self).__init__()

    def _m_c2deflections(self, logM, concentration):
        """

        :param logM: log10 mass in M200 stellar masses
        :param concentration: halo concentration c = r_200 / r_s
        :return: Rs (in arc seconds), alpha_Rs (in arc seconds)
        """
        if self._static is True:
            return self._Rs_static, self._alpha_Rs_static
        M = 10 ** logM
        Rs, alpha_Rs = self._lens_cosmo.nfw_physical2angle(M, concentration)
        return Rs, alpha_Rs

    def set_static(self, logM, concentration, center_x=0, center_y=0):
        """

        :param logM:
        :param concentration:
        :param center_x:
        :param center_y:
        :return:
        """
        self._static = True
        M = 10 ** logM
        self._Rs_static, self._alpha_Rs_static = self._lens_cosmo.nfw_physical2angle(M, concentration)

    def set_dynamic(self):
        """

        :return:
        """
        self._static = False
        if hasattr(self, '_Rs_static'):
            del self._Rs_static
        if hasattr(self, '_alpha_Rs_static'):
            del self._alpha_Rs_static

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
        returns Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx, d^f/dy^2
        """
        Rs, alpha_Rs = self._m_c2deflections(logM, concentration)
        return self._nfw.hessian(x, y, Rs, alpha_Rs, center_x, center_y)
