__author__ = 'nataliehogg'

import numpy as np
import numpy.testing as npt
import unittest
import pytest

from lenstronomy.LensModel.LineOfSight.single_plane_los import SinglePlaneLOS
from lenstronomy.LensModel.Profiles.sis import SIS

class TestSinglePlaneLOS(object):
    """
    tests the SinglePlaneLOS routines

    these functions are the same as in TestLensModel but with the addition of LOS and LOS_MINIMAL as profiles.
    with all params in self.kwargs_los set to zero, the results should be the same as the non-LOS cases originally tested
    """

    def setup(self):
        self.lensModel = SinglePlaneLOS(['GAUSSIAN', 'LOS'])
        self.lensModel_minimal = SinglePlaneLOS(['GAUSSIAN', 'LOS_MINIMAL'])
        self.kwargs = {'amp': 1., 'sigma_x': 2., 'sigma_y': 2., 'center_x': 0., 'center_y': 0.}
        self.los_kwargs = {'gamma1_os': 0.0, 'gamma2_os': 0.0, 'kappa_os': 0.0, 'omega_os': 0.0,
                           'gamma1_od': 0.0, 'gamma2_od': 0.0, 'kappa_od': 0.0, 'omega_od': 0.0,
                           'gamma1_ds': 0.0, 'gamma2_ds': 0.0, 'kappa_ds': 0.0, 'omega_ds': 0.0}

    def test_potential(self):
        output = self.lensModel.potential(x=1., y=1., kwargs=[self.kwargs, self.los_kwargs])
        output_minimal = self.lensModel_minimal.potential(x=1., y=1., kwargs=[self.kwargs, self.los_kwargs])
        assert output == 0.77880078307140488/(8*np.pi)
        assert output_minimal == 0.77880078307140488/(8*np.pi)

    def test_alpha(self):
        output1, output2 = self.lensModel.alpha(x=1., y=1., kwargs=[self.kwargs, self.los_kwargs])
        output1_minimal, output2_minimal = self.lensModel_minimal.alpha(x=1., y=1., kwargs=[self.kwargs, self.los_kwargs])
        assert output1 == -0.19470019576785122/(8*np.pi)
        assert output2 == -0.19470019576785122/(8*np.pi)
        assert output1_minimal == -0.19470019576785122/(8*np.pi)
        assert output2_minimal == -0.19470019576785122/(8*np.pi)

    def test_ray_shooting(self):
        delta_x, delta_y = self.lensModel.ray_shooting(x=1., y=1., kwargs=[self.kwargs, self.los_kwargs])
        delta_x_minimal, delta_y_minimal = self.lensModel_minimal.ray_shooting(x=1., y=1., kwargs=[self.kwargs, self.los_kwargs])
        assert delta_x == 1 + 0.19470019576785122/(8*np.pi)
        assert delta_y == 1 + 0.19470019576785122/(8*np.pi)
        assert delta_x_minimal == 1 + 0.19470019576785122/(8*np.pi)
        assert delta_y_minimal == 1 + 0.19470019576785122/(8*np.pi)

    def test_mass_2d(self):
        lensModel = SinglePlaneLOS(['GAUSSIAN_KAPPA', 'LOS'])
        lensModel_minimal = SinglePlaneLOS(['GAUSSIAN_KAPPA', 'LOS_MINIMAL'])
        output = lensModel.mass_2d(r=1, kwargs=[self.kwargs, self.los_kwargs])
        output_minimal = lensModel_minimal.mass_2d(r=1, kwargs=[self.kwargs, self.los_kwargs])
        assert output == 0.11750309741540453
        assert output_minimal == 0.11750309741540453

    def test_density(self):
        theta_E = 1
        r = 1
        sis = SIS()
        density_model = sis.density_lens(r=r, theta_E=theta_E)

        # LOS
        lensModel = SinglePlaneLOS(lens_model_list=['SIS', 'LOS'])
        density = lensModel.density(r=r, kwargs=[{'theta_E': theta_E}, self.los_kwargs])
        npt.assert_almost_equal(density, density_model, decimal=8)

        # LOS_MINIMAL
        lensModel_minimal = SinglePlaneLOS(lens_model_list = ['SIS', 'LOS_MINIMAL'])
        density_minimal = lensModel_minimal.density(r=r, kwargs=[{'theta_E': theta_E}, self.los_kwargs])
        npt.assert_almost_equal(density_minimal, density_model, decimal=8)

    def test_bool_list(self):
        lensModel = SinglePlaneLOS(['SPEP', 'SHEAR', 'LOS'])
        lensModel_minimal = SinglePlaneLOS(['SPEP', 'SHEAR', 'LOS_MINIMAL'])
        kwargs = [{'theta_E': 1, 'gamma': 2, 'e1': 0.1, 'e2': -0.1, 'center_x': 0, 'center_y': 0},
                           {'gamma1': 0.01, 'gamma2': -0.02}, self.los_kwargs]

        # LOS
        alphax_1, alphay_1 = lensModel.alpha(1, 1, kwargs, k=0)
        alphax_1_list, alphay_1_list = lensModel.alpha(1, 1, kwargs, k=[0])
        npt.assert_almost_equal(alphax_1, alphax_1_list, decimal=5)
        npt.assert_almost_equal(alphay_1, alphay_1_list, decimal=5)

        alphax_1_1, alphay_1_1 = lensModel.alpha(1, 1, kwargs, k=0)
        alphax_1_2, alphay_1_2 = lensModel.alpha(1, 1, kwargs, k=1)
        alphax_full, alphay_full = lensModel.alpha(1, 1, kwargs, k=None)
        npt.assert_almost_equal(alphax_1_1 + alphax_1_2, alphax_full, decimal=5)
        npt.assert_almost_equal(alphay_1_1 + alphay_1_2, alphay_full, decimal=5)

        # LOS_MINIMAL
        alphax_1_minimal, alphay_1_minimal = lensModel_minimal.alpha(1, 1, kwargs, k=0)
        alphax_1_list_minimal, alphay_1_list_minimal = lensModel_minimal.alpha(1, 1, kwarhs, k=[0])
        npt.assert_almost_equal(alphax_1_minimal, alphax_1_list_minimal, decimal=5)
        npt.assert_almost_equal(alphay_1_minimal, alphay_1_list_minimal, decimal=5)

        alphax_1_1_minimal, alphay_1_1_minimal = lensModel_minimal.alpha(1, 1, kwargs, k=0)
        alphax_1_2_minimal, alphay_1_2_minimal = lensModel_minimal.alpha(1, 1, kwargs, k=1)
        alphax_full_minimal, alphay_full_minimal = lensModel_minimal.alpha(1, 1, kwargs, k=None)
        npt.assert_almost_equal(alphax_1_1_minimal + alphax_1_2_minimal, alphax_full_minimal, decimal=5)
        npt.assert_almost_equal(alphay_1_1_minimal + alphay_1_2_minimal, alphay_full_minimal, decimal=5)

    def test_init(self):
        lens_model_list = ['TNFW', 'TRIPLE_CHAMELEON', 'SHEAR_GAMMA_PSI', 'CURVED_ARC_CONST',
                           'NFW_MC', 'ARC_PERT', 'MULTIPOLE', 'CURVED_ARC_SPP']
        lensModel = SinglePlaneLOS(lens_model_list=lens_model_list)
        assert lensModel.func_list[0].param_names[0] == 'Rs'

class TestRaise(unittest.TestCase):

    def test_raise(self):
        """
        check whether raises occurs if fastell4py is not installed

        :return:
        """
        if bool_test is False:
            with self.assertRaises(ImportError):
                SinglePlane(lens_model_list=['PEMD'])
            with self.assertRaises(ImportError):
                SinglePlane(lens_model_list=['SPEMD'])
        else:
            SinglePlane(lens_model_list=['PEMD', 'SPEMD'])


if __name__ == '__main__':
    pytest.main("-k TestLensModel")
