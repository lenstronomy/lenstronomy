import numpy as np
import lenstronomy.Util.constants as const
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from astropy.cosmology import FlatLambdaCDM

__all__ = ['Uldm_m_Likelihood']


class Uldm_m_Likelihood(object):
    """
    class to compute the likelihood of mass of ULDM given angular variables from ULDM-BAR Lens Profile,
    it is used mostly as a prior
    """
    def __init__(self, ULDM_mass_fixed, ULDM_sigma_mass, lens_model_class, point_source_class):
        """

        :param mass: log10(mass), with mass the mass of the ULDM particle in eV
        :param sigma: uncertainty given to the range of masses one can assume (in the module, it is multiplied by 10**(mass) in order to have a sigma of the order of the mass; choose ULDM_sigma_mass of the order of 0.01 to have a sharp prior)
        :param lens_model_class: instance of the LensModel() class
        :param point_source_class: instance of the PointSource() class, note: the first point source type is the one the
        time delays are imposed on
        """

        if lens_model_class.lens_model_list[0] != 'ULDM-BAR':
            raise ValueError("lens model need to be ULDM-BAR to evaluate the Uldm_m_Likelihood.")
        self._ULDM_mass_fixed = 10**(ULDM_mass_fixed)
        self._ULDM_sigma_mass = ULDM_sigma_mass * 10**(ULDM_mass_fixed)
        self._lensModel = lens_model_class
        self._pointSource = point_source_class

    def logL(self, kwargs_lens, kwargs_ps, kwargs_cosmo):
        """
        routine to compute the log likelihood of the time delay distance (it assumes
        that ULDM-BAR is the first lens model!)
        :param kwargs_lens: lens model kwargs list of ULDM-BAR class
        :param kwargs_ps: point source kwargs list
        :param kwargs_cosmo: cosmology and other kwargs
        :return: log likelihood of the model given the time delay data
        """
        z_lens = kwargs_cosmo['z_lens']
        z_source = kwargs_cosmo['z_source']
        Om = kwargs_cosmo['Om']
        h0_model = kwargs_cosmo['h0']
        cosmo = FlatLambdaCDM(H0 = h0_model, Om0=Om, Ob0=0.)
        lens_cosmo = LensCosmo(z_lens = z_lens, z_source = z_source, cosmo = cosmo)
        kappa_0 = kwargs_lens[0]['kappa_0']
        theta_c = kwargs_lens[0]['theta_c']
        theta_E = kwargs_lens[0]['theta_E']
        mass, Mass, rho0, lambda_factor = lens_cosmo.ULDM_BAR_angles2phys(kappa_0, theta_c, theta_E)
        mass = 10**(mass)
        logL = -(mass - self._ULDM_mass_fixed) ** 2 / (2 * self._ULDM_sigma_mass ** 2)
        return logL
    @property
    def num_data(self):
        """

        :return: number of time delay measurements
        """
        return 1
