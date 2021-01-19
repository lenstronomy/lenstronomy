from lenstronomy.GalKin.numeric_kinematics import NumericKinematics
from lenstronomy.GalKin.analytic_kinematics import AnalyticKinematics

__all__ = ['GalkinModel']


class GalkinModel(object):
    """
    this class handles all the kinematic modeling aspects of Galkin
    Excluded are observational conditions (seeing, aperture etc)
    Major class to compute velocity dispersion measurements given light and mass models

    The class supports any mass and light distribution (and superposition thereof) that has a 3d correspondance in their
    2d lens model distribution. For models that do not have this correspondence, you may want to apply a
    Multi-Gaussian Expansion (MGE) on their models and use the MGE to be de-projected to 3d.

    The computation follows Mamon&Lokas 2005.

    The class supports various types of anisotropy models (see Anisotropy class).
    Solving the Jeans Equation requires a numerical integral over the 3d light and mass profile (see Mamon&Lokas 2005).
    This class (as well as the dedicated LightModel and MassModel classes) perform those integral numerically with an
    interpolated grid.

    The cosmology assumed to compute the physical mass and distances are set via the kwargs_cosmo keyword arguments.
        d_d: Angular diameter distance to the deflector (in Mpc)
        d_s: Angular diameter distance to the source (in Mpc)
        d_ds: Angular diameter distance from the deflector to the source (in Mpc)

    The numerical options can be chosen through the kwargs_numerics keywords

        interpol_grid_num: number of interpolation points in the light and mass profile (radially). This number should
        be chosen high enough to accurately describe the true light profile underneath.
        log_integration: bool, if True, performs the interpolation and numerical integration in log-scale.

        max_integrate: maximum 3d radius to where the numerical integration of the Jeans Equation solver is made.
        This value should be large enough to contain most of the light and to lead to a converged result.
        min_integrate: minimal integration value. This value should be very close to zero but some mass and light
        profiles are diverging and a numerically stable value should be chosen.

    These numerical options should be chosen to allow for a converged result (within your tolerance) but not too
    conservative to impact too much the computational cost. Reasonable values might depend on the specific problem.

    """
    def __init__(self, kwargs_model, kwargs_cosmo, kwargs_numerics=None, analytic_kinematics=False):
        """

        :param kwargs_model: keyword arguments describing the model components
        :param kwargs_cosmo: keyword arguments that define the cosmology in terms of the angular diameter distances involved
        :param kwargs_numerics: numerics keyword arguments
        :param analytic_kinematics: bool, if True uses the analytic kinematic model
        """
        if kwargs_numerics is None:
            kwargs_numerics = {'interpol_grid_num': 200,  # numerical interpolation, should converge -> infinity
                               'log_integration': True,
                               # log or linear interpolation of surface brightness and mass models
                               'max_integrate': 100,
                               'min_integrate': 0.001}  # lower/upper bound of numerical integrals
        if analytic_kinematics is True:
            anisotropy_model = kwargs_model.get('anisotropy_model')
            if not anisotropy_model == 'OM':
                raise ValueError('analytic kinematics only available for OsipkovMerritt ("OM") anisotropy model.')
            self.numerics = AnalyticKinematics(kwargs_cosmo=kwargs_cosmo, **kwargs_numerics)
        else:
            self.numerics = NumericKinematics(kwargs_model=kwargs_model, kwargs_cosmo=kwargs_cosmo, **kwargs_numerics)
        self._analytic_kinematics = analytic_kinematics

    def check_df(self, r, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        checks whether the phase space distribution function of a given anisotropy model is positive.
        Currently this is implemented by the relation provided by Ciotti and Morganti 2010 equation (10)
        https://arxiv.org/pdf/1006.2344.pdf

        :param r: 3d radius to check slope-anisotropy constraint
        :param theta_E: Einstein radius in arc seconds
        :param gamma: power-law slope
        :param a_ani: scaled transition radius of the OM anisotropy distribution
        :param r_eff: half-light radius in arc seconds
        :return: equation (10) >= 0 for physical interpretation
        """
        dr = 0.01  # finite differential in radial direction
        r_dr = r + dr

        sigmar2 = self.numerics.sigma_r2(r, kwargs_mass, kwargs_light, kwargs_anisotropy)
        sigmar2_dr = self.numerics.sigma_r2(r_dr, kwargs_mass, kwargs_light, kwargs_anisotropy)
        grav_pot = self.numerics.grav_potential(r, kwargs_mass)
        grav_pot_dr = self.numerics.grav_potential(r_dr, kwargs_mass)
        self.numerics.delete_cache()
        return r * (sigmar2_dr - sigmar2 - grav_pot + grav_pot_dr) / dr
