__author__ = 'sibirrer'

import pytest
import numpy.testing as npt
import numpy as np

from lenstronomy.LensModel.profile_integrals import ProfileIntegrals


class TestNumerics(object):
    """
    tests the second derivatives of various lens models
    """
    def setup(self):
        pass

    def assert_integrals(self, Model, kwargs):
        lensModel = Model()
        int_profile = ProfileIntegrals(lensModel)
        r = 2.

        density2d_num = int_profile.density_2d(r, kwargs)
        density2d = lensModel.density_2d(r, 0, **kwargs)
        npt.assert_almost_equal(density2d/density2d_num, 1., decimal=1)

        kwargs['center_x'] = 0
        kwargs['center_y'] = 0
        mass_2d_num = int_profile.mass_enclosed_2d(r, kwargs)
        del kwargs['center_x']
        del kwargs['center_y']
        mass_2d = lensModel.mass_2d(r, **kwargs)
        npt.assert_almost_equal(mass_2d/mass_2d_num, 1, decimal=1)

        kwargs['center_x'] = 0
        kwargs['center_y'] = 0
        mass_3d_num = int_profile.mass_enclosed_3d(r, kwargs)
        del kwargs['center_x']
        del kwargs['center_y']
        mass_3d = lensModel.mass_3d(r, **kwargs)
        npt.assert_almost_equal(mass_3d/mass_3d_num, 1, decimal=2)

    def assert_lens_integrals(self, Model, kwargs):
        """
        checks whether the integral in projection of the density_lens() function is the convergence

        :param Model: lens model instance
        :param kwargs: keyword arguments of lens model
        :return:
        """
        lensModel = Model()
        int_profile = ProfileIntegrals(lensModel)
        r = 2.
        kappa_num = int_profile.density_2d(r, kwargs, lens_param=True)
        f_xx, f_yy, f_xy = lensModel.hessian(r, 0, **kwargs)
        kappa = 1./2 * (f_xx + f_yy)
        npt.assert_almost_equal(kappa_num, kappa, decimal=2)
        if hasattr(lensModel, 'mass_2d_lens'):
            mass_2d = lensModel.mass_2d_lens(r, **kwargs)
            alpha_x, alpha_y = lensModel.derivatives(r, 0, **kwargs)
            alpha = np.sqrt(alpha_x**2 + alpha_y**2)
            npt.assert_almost_equal(alpha, mass_2d/ r / np.pi, decimal=5)

    def test_PJaffe(self):
        kwargs = {'rho0': 1., 'Ra': 0.2, 'Rs': 2.}
        from lenstronomy.LensModel.Profiles.p_jaffe import PJaffe as Model
        self.assert_integrals(Model, kwargs)

    def test_PJaffa_density_deflection(self):
        """
        tests whether the unit conversion between the lensing parameter 'sigma0' and the units in the density profile are ok
        :return:
        """

        from lenstronomy.LensModel.Profiles.p_jaffe import PJaffe as Model
        lensModel = Model()
        sigma0 = 1.
        Ra = 0.2
        Rs = 2.
        rho0 = lensModel.sigma2rho(sigma0, Ra, Rs)
        kwargs_lens = {'sigma0': sigma0, 'Ra': Ra, 'Rs': Rs, 'center_x': 0, 'center_y': 0}
        kwargs_density = {'rho0': rho0, 'Ra': Ra, 'Rs': Rs}
        r = 1.
        mass_2d = lensModel.mass_2d(r, **kwargs_density)
        alpha_mass = mass_2d/r
        alpha_r, _ = lensModel.derivatives(r, 0, **kwargs_lens)
        npt.assert_almost_equal(alpha_mass/np.pi, alpha_r, decimal=5)

    def test_nfw(self):
        kwargs = {'rho0': 1.,  'Rs': 5., 'center_x': 0, 'center_y': 0}
        from lenstronomy.LensModel.Profiles.nfw import NFW as Model
        self.assert_integrals(Model, kwargs)

        kwargs_lens = {'alpha_Rs': 1., 'Rs': 5., 'center_x': 0, 'center_y': 0}
        self.assert_lens_integrals(Model, kwargs_lens)

    def test_nfw_ellipse(self):
        kwargs = {'rho0': 1.,  'Rs': 5., 'center_x': 0, 'center_y': 0, 'e1': 0, 'e2': 0}
        from lenstronomy.LensModel.Profiles.nfw_ellipse import NFW_ELLIPSE as Model
        #self.assert_integrals(Model, kwargs)

        kwargs_lens = {'alpha_Rs': 1., 'Rs': 5., 'center_x': 0, 'center_y': 0, 'e1': 0, 'e2': 0}
        self.assert_lens_integrals(Model, kwargs_lens)

    def test_nfw_density_deflection(self):
        """
        tests whether the unit conversion between the lensing parameter 'sigma0' and the units in the density profile are ok
        :return:
        """

        from lenstronomy.LensModel.Profiles.nfw import NFW as Model
        lensModel = Model()
        alpha_Rs = 1.
        Rs = 2.
        rho0 = lensModel._alpha2rho0(alpha_Rs, Rs)
        kwargs_lens = {'alpha_Rs': alpha_Rs, 'Rs': Rs}
        kwargs_density = {'rho0': rho0, 'Rs': Rs}
        r = 2.
        mass_2d = lensModel.mass_2d(r, **kwargs_density)
        alpha_mass = mass_2d/r
        alpha_r, _ = lensModel.derivatives(r, 0, **kwargs_lens)
        npt.assert_almost_equal(alpha_mass/np.pi, alpha_r, decimal=5)

    def test_hernquist(self):
        from lenstronomy.LensModel.Profiles.hernquist import Hernquist as Model
        kwargs = {'rho0': 1., 'Rs': 5.}
        self.assert_integrals(Model, kwargs)
        kwargs = {'sigma0': 1., 'Rs': 5.}
        self.assert_lens_integrals(Model, kwargs)

    def test_hernquist_ellipse(self):
        from lenstronomy.LensModel.Profiles.hernquist_ellipse import Hernquist_Ellipse as Model
        kwargs = {'rho0': 1., 'Rs': 5., 'e1': 0, 'e2': 0}
        self.assert_integrals(Model, kwargs)
        kwargs = {'sigma0': 1., 'Rs': 5., 'e1': 0, 'e2': 0}
        self.assert_lens_integrals(Model, kwargs)

    def test_hernquist_density_deflection(self):
        """
        tests whether the unit conversion between the lensing parameter 'sigma0' and the units in the density profile are ok
        :return:
        """

        from lenstronomy.LensModel.Profiles.hernquist import Hernquist as Model
        lensModel = Model()
        sigma0 = 1.
        Rs = 2.
        rho0 = lensModel.sigma2rho(sigma0, Rs)
        kwargs_lens = {'sigma0': sigma0, 'Rs': Rs}
        kwargs_density = {'rho0': rho0, 'Rs': Rs}
        r = .5
        mass_2d = lensModel.mass_2d(r, **kwargs_density)
        alpha_mass = mass_2d/r
        alpha_r, _ = lensModel.derivatives(r, 0, **kwargs_lens)
        npt.assert_almost_equal(alpha_mass/np.pi, alpha_r, decimal=5)

    def test_sis(self):
        from lenstronomy.LensModel.Profiles.sis import SIS as Model
        kwargs = {'rho0': 1.}
        self.assert_integrals(Model, kwargs)
        kwargs_lens = {'theta_E': 1.}
        self.assert_lens_integrals(Model, kwargs_lens)

    def test_sis_density_deflection(self):
        """
        tests whether the unit conversion between the lensing parameter 'sigma0' and the units in the density profile are ok
        :return:
        """

        from lenstronomy.LensModel.Profiles.sis import SIS as Model
        lensModel = Model()
        theta_E = 1.
        rho0 = lensModel.theta2rho(theta_E)
        kwargs_lens = {'theta_E': theta_E}
        kwargs_density = {'rho0': rho0}
        r = .5
        mass_2d = lensModel.mass_2d(r, **kwargs_density)
        alpha_mass = mass_2d/r
        alpha_r, _ = lensModel.derivatives(r, 0, **kwargs_lens)
        npt.assert_almost_equal(alpha_mass/np.pi, alpha_r, decimal=5)
        lensModel.density_2d(1, 1, rho0=1)

    def test_sie(self):
        from lenstronomy.LensModel.Profiles.sie import SIE as Model
        kwargs = {'rho0': 1., 'e1': 0, 'e2': 0}
        self.assert_integrals(Model, kwargs)
        kwargs_lens = {'theta_E': 1., 'e1': 0, 'e2': 0}
        self.assert_lens_integrals(Model, kwargs_lens)

    def test_spep(self):
        from lenstronomy.LensModel.Profiles.spep import SPEP as Model
        kwargs_lens = {'theta_E': 1, 'gamma': 2, 'e1': 0, 'e2': 0}
        self.assert_lens_integrals(Model, kwargs_lens)

    def test_sie_density_deflection(self):
        """
        tests whether the unit conversion between the lensing parameter 'sigma0' and the units in the density profile are ok
        :return:
        """

        from lenstronomy.LensModel.Profiles.sie import SIE as Model
        lensModel = Model()
        theta_E = 1.
        rho0 = lensModel.theta2rho(theta_E)
        kwargs_lens = {'theta_E': theta_E, 'e1': 0, 'e2': 0}
        kwargs_density = {'rho0': rho0, 'e1': 0, 'e2': 0}
        r = .5
        mass_2d = lensModel.mass_2d(r, **kwargs_density)
        alpha_mass = mass_2d/r
        alpha_r, _ = lensModel.derivatives(r, 0, **kwargs_lens)
        npt.assert_almost_equal(alpha_mass/np.pi, alpha_r, decimal=5)

    def test_spp(self):
        from lenstronomy.LensModel.Profiles.spp import SPP as Model
        kwargs = {'rho0': 10., 'gamma': 2.2}
        self.assert_integrals(Model, kwargs)

        kwargs = {'rho0': 1., 'gamma': 2.0}
        self.assert_integrals(Model, kwargs)
        kwargs_lens = {'theta_E': 1., 'gamma': 2.0}
        self.assert_lens_integrals(Model, kwargs_lens)

    def test_spp_density_deflection(self):
        """
        tests whether the unit conversion between the lensing parameter 'sigma0' and the units in the density profile are ok
        :return:
        """

        from lenstronomy.LensModel.Profiles.spp import SPP as Model
        lensModel = Model()
        theta_E = 1.
        gamma = 2.2
        rho0 = lensModel.theta2rho(theta_E, gamma)
        kwargs_lens = {'theta_E': theta_E, 'gamma': gamma}
        kwargs_density = {'rho0': rho0, 'gamma': gamma}
        r = .5
        mass_2d = lensModel.mass_2d(r, **kwargs_density)
        alpha_mass = mass_2d/r
        alpha_r, _ = lensModel.derivatives(r, 0, **kwargs_lens)
        npt.assert_almost_equal(alpha_mass/np.pi, alpha_r, decimal=5)

    def test_gaussian(self):
        from lenstronomy.LensModel.Profiles.gaussian_kappa import GaussianKappa as Model
        kwargs = {'amp': 1. / 4., 'sigma': 2.}
        self.assert_integrals(Model, kwargs)

    def test_gaussian_density_deflection(self):
        """
        tests whether the unit conversion between the lensing parameter 'sigma0' and the units in the density profile are ok
        :return:
        """

        from lenstronomy.LensModel.Profiles.gaussian_kappa import GaussianKappa as Model
        lensModel = Model()
        amp = 1. / 4.
        sigma = 2.
        amp_lens = lensModel._amp3d_to_2d(amp, sigma, sigma)
        kwargs_lens = {'amp': amp_lens, 'sigma': sigma}
        kwargs_density = {'amp': amp, 'sigma': sigma}
        r = .5
        mass_2d = lensModel.mass_2d(r, **kwargs_density)
        alpha_mass = mass_2d/r
        alpha_r, _ = lensModel.derivatives(r, 0, **kwargs_lens)
        npt.assert_almost_equal(alpha_mass/np.pi, alpha_r, decimal=5)

    def test_coreBurk(self):
        from lenstronomy.LensModel.Profiles.coreBurkert import CoreBurkert as Model
        kwargs = {'rho0': 1., 'Rs': 10, 'r_core': 5}
        self.assert_integrals(Model, kwargs)

        kwargs = {'rho0': 1., 'Rs': 9, 'r_core': 11}
        self.assert_integrals(Model, kwargs)

    def test_tnfw(self):
        from lenstronomy.LensModel.Profiles.tnfw import TNFW as Model
        kwargs = {'rho0': 1., 'Rs': 1, 'r_trunc': 4}
        self.assert_integrals(Model, kwargs)

    def test_cnfw(self):
        from lenstronomy.LensModel.Profiles.cnfw import CNFW as Model
        kwargs = {'rho0': 1., 'Rs': 1, 'r_core': 0.5}
        self.assert_integrals(Model, kwargs)
        kwargs_lens = {'alpha_Rs': 1., 'Rs': 5., 'r_core': 0.5}
        self.assert_lens_integrals(Model, kwargs_lens)

    def test_cnfw_ellipse(self):
        from lenstronomy.LensModel.Profiles.cnfw_ellipse import CNFW_ELLIPSE as Model
        kwargs = {'rho0': 1., 'Rs': 1, 'r_core': 0.5, 'e1': 0, 'e2':0}
        #self.assert_integrals(Model, kwargs)
        kwargs_lens = {'alpha_Rs': 1., 'Rs': 5., 'r_core': 0.5, 'e1': 0, 'e2':0}
        self.assert_lens_integrals(Model, kwargs_lens)

    def test_cored_density(self):
        from lenstronomy.LensModel.Profiles.cored_density import CoredDensity as Model
        kwargs = {'sigma0': 0.1, 'r_core': 6.}
        self.assert_integrals(Model, kwargs)
        self.assert_lens_integrals(Model, kwargs)

    def test_cored_density_2(self):
        from lenstronomy.LensModel.Profiles.cored_density_2 import CoredDensity2 as Model
        kwargs = {'sigma0': 0.1, 'r_core': 6.}
        self.assert_integrals(Model, kwargs)
        self.assert_lens_integrals(Model, kwargs)

    def test_cored_density_exp(self):
        from lenstronomy.LensModel.Profiles.cored_density_exp import CoredDensityExp as Model
        kwargs = {'kappa_0': 0.1, 'theta_c': 6.}
        self.assert_integrals(Model, kwargs)
        self.assert_lens_integrals(Model, kwargs)

    def test_uldm(self):
        from lenstronomy.LensModel.Profiles.uldm import Uldm as Model
        kwargs = {'kappa_0': 0.1, 'theta_c': 6.}
        self.assert_integrals(Model, kwargs)
        self.assert_lens_integrals(Model, kwargs)

    def test_splcore(self):

        from lenstronomy.LensModel.Profiles.splcore import SPLCORE as Model
        kwargs = {'rho0': 1., 'gamma': 3, 'r_core': 0.1}
        self.assert_integrals(Model, kwargs)
        kwargs = {'sigma0': 1., 'gamma': 3, 'r_core': 0.1}
        self.assert_lens_integrals(Model, kwargs)

        kwargs = {'rho0': 1., 'gamma': 2, 'r_core': 0.1}
        self.assert_integrals(Model, kwargs)
        kwargs = {'sigma0': 1., 'gamma': 2, 'r_core': 0.1}
        self.assert_lens_integrals(Model, kwargs)

        kwargs = {'rho0': 1., 'gamma': 2.5, 'r_core': 0.1}
        self.assert_integrals(Model, kwargs)
        kwargs = {'sigma0': 1., 'gamma': 2.5, 'r_core': 0.1}
        self.assert_lens_integrals(Model, kwargs)


    """
    def test_sis(self):
        kwargs = {'theta_E': 0.5}
        from astrofunc.LensingProfiles.sis import SIS as Model
        self.assert_integrals(Model, kwargs)

    def test_sersic(self):
        kwargs = {'n_sersic': .5, 'r_eff': 1.5, 'k_eff': 0.3}
        from astrofunc.LensingProfiles.sersic import Sersic as Model
        self.assert_integrals(Model, kwargs)

    """


if __name__ == '__main__':
    pytest.main("-k TestLensModel")
