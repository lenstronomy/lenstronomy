import numpy as np
import lenstronomy.Util.constants as const
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Analysis.kinematics_api import KinematicsAPI
__all__ = ['KinLikelihood']


class KinLikelihood(object):
    """
    class to compute the kinetic likelihood of a model given a measurement of velocity dispersions
    """
    def __init__(self, sigma_measured, sigma_uncertainties, lens_model_class, point_source_class):
        """
        :param time_delays_measured: relative time delays (in days) in respect to the first image of the point source
        :param time_delays_uncertainties: time-delay uncertainties in same order as time_delay_measured
        :param lens_model_class: instance of the LensModel() class
        :param point_source_class: instance of the PointSource() class, note: the first point source type is the one the
        time delays are imposed on
        """

        if sigma_measured is None:
            raise ValueError("sigma_measured need to be specified to evaluate the kinetic likelihood.")
        if sigma_uncertainties is None:
            raise ValueError("sigma_uncertainties need to be specified to evaluate the kinetic likelihood.")
        self._sigma_measured = np.array(sigma_measured)
        self._sigma_errors = np.array(sigma_uncertainties)
        self._lensModel = lens_model_class
        self._pointSource = point_source_class

    def logL(self, kwargs_lens, kwargs_ps, kwargs_cosmo):
        """
        routine to compute the log likelihood of kinematics
        :param kwargs_lens: lens model kwargs list
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
        ### put all this as input in kwargs_something
        R_slit = 1. # slit length in arcsec
        dR_slit = 1.  # slit width in arcsec
        psf_fwhm = 0.7

        kwargs_aperture = {'aperture_type': 'slit', 'length': R_slit, 'width': dR_slit, 'center_ra': 0.05, 'center_dec': 0, 'angle': 0}
        anisotropy_model = 'OM' # Osipkov-Merritt
        aperture_type = 'slit'
        kwargs_numerics_galkin = {'interpol_grid_num': 1000,  # numerical interpolation, should converge -> infinity
                                  'log_integration': True,  # log or linear interpolation of surface brightness and mass models
                                   'max_integrate': 100, 'min_integrate': 0.001}  # lower/upper bound of numerical integrals
        r_ani = 1.
        r_eff = 0.2
        kwargs_anisotropy = {'r_ani': r_ani}
        kwargs_seeing = {'psf_type': 'GAUSSIAN', 'fwhm': psf_fwhm}

        kin_api = KinematicsAPI(z_lens, z_source, kwargs_lens, cosmo=cosmo,
                                lens_model_kinematics_bool=[True, False], light_model_kinematics_bool=[True],
                                kwargs_aperture=kwargs_aperture, kwargs_seeing=kwargs_seeing,
                                anisotropy_model=anisotropy_model, kwargs_numerics_galkin=kwargs_numerics_galkin,
                                sampling_number=10000,  # numerical ray-shooting, should converge -> infinity
                                Hernquist_approx=True)

        vel_disp = kin_api.velocity_dispersion(kwargs_lens, kwargs_lens_light, kwargs_anisotropy, r_eff=r_eff, theta_E=None, kappa_ext=0)
        logL = np.sum(-(vel_disp - self._sigma_measured)**2 / (2 * self._sigma_errors**2))
        return logL

