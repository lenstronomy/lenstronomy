import pytest
import numpy as np
import numpy.testing as npt
from lenstronomy.LensModel.QuadOptimizer.param_manager import PowerLawFixedShear, \
    PowerLawFixedShearMultipole, PowerLawFreeShear, PowerLawFreeShearMultipole

class TestParamClasses(object):

    def setup(self):

        self.zlens, self.zsource = 0.5, 1.5
        epl_kwargs = {'theta_E': 1., 'center_x': 0., 'center_y': 0., 'e1': 0.2, 'e2': 0.1, 'gamma': 2.05}
        shear_kwargs = {'gamma1': 0.05, 'gamma2': -0.04}
        kwargs_macro = [epl_kwargs, shear_kwargs]

        self.x_image = np.array([0.65043538, -0.31109505, 0.78906059, -0.86222271])
        self.y_image = np.array([-0.89067493, 0.94851787, 0.52882605, -0.25403778])

        halo_list = ['SIS', 'SIS', 'SIS']
        halo_z = [self.zlens - 0.1, self.zlens, self.zlens + 0.4]
        halo_kwargs = [{'theta_E': 0.1, 'center_x': 0.3, 'center_y': -0.9},
                       {'theta_E': 0.15, 'center_x': 1.3, 'center_y': -0.5},
                       {'theta_E': 0.06, 'center_x': -0.4, 'center_y': -0.4}]

        self.kwargs_epl = kwargs_macro + halo_kwargs
        self.zlist_epl = [self.zlens, self.zlens] + halo_z
        self.lens_model_list_epl = ['EPL', 'SHEAR'] + halo_list

        kwargs_multi = [{'m': 4, 'a_m': -0.04, 'phi_m': -0.2, 'center_x': 0.1, 'center_y': -0.1}]
        self.kwargs_multipole = kwargs_macro + kwargs_multi + halo_kwargs
        self.zlist_multipole = [self.zlens, self.zlens, self.zlens] + halo_z
        self.lens_model_list_multipole = ['EPL', 'SHEAR'] + ['MULTIPOLE'] + halo_list

    def test_plaw_free_shear(self):

        param_class = PowerLawFreeShear(self.kwargs_epl)
        npt.assert_(param_class.to_vary_index==2)
        kwargs_in = [{'theta_E': 1., 'center_x': 0., 'center_y': 0.3, 'e1': 0.25, 'e2': 0.1, 'gamma': 2.05},
                     {'gamma1': 0.05, 'gamma2': -0.01}, {'theta_E': -0.3, 'center_x': 0., 'center_y': 0.04}]
        args_epl = param_class.kwargs_to_args(kwargs_in)
        npt.assert_almost_equal(args_epl, [1, 0, 0.3, 0.25, 0.1, 0.05, -0.01])
        kwargs_out = param_class.args_to_kwargs(args_epl)
        npt.assert_almost_equal(kwargs_out[0]['gamma'], 2.05)
        for key in kwargs_out[-1].keys():
            npt.assert_almost_equal(kwargs_out[-1][key], self.kwargs_epl[-1][key])

    def test_plaw_fixed_shear(self):
        param_class = PowerLawFixedShear(self.kwargs_epl, 0.12)
        npt.assert_(param_class.to_vary_index == 2)
        kwargs_in = [{'theta_E': 1., 'center_x': 0., 'center_y': 0.3, 'e1': 0.25, 'e2': 0.1, 'gamma': 2.05},
                     {'gamma1': 0.05, 'gamma2': -0.01}, {'theta_E': -0.3, 'center_x': 0., 'center_y': 0.04}]
        args_epl = param_class.kwargs_to_args(kwargs_in)
        npt.assert_almost_equal(args_epl[0:5], [1, 0, 0.3, 0.25, 0.1])
        kwargs_out = param_class.args_to_kwargs(args_epl)
        npt.assert_almost_equal(kwargs_out[0]['gamma'], 2.05)
        npt.assert_almost_equal(kwargs_out[1]['gamma1'] ** 2 + kwargs_out[1]['gamma2']**2, 0.12 ** 2)
        for key in kwargs_out[-1].keys():
            npt.assert_almost_equal(kwargs_out[-1][key], self.kwargs_epl[-1][key])

    def test_plawboxydisky_fixed_shear(self):

        param_class = PowerLawFixedShearMultipole(self.kwargs_multipole, 0.12)
        npt.assert_(param_class.to_vary_index == 3)
        kwargs_in = [{'theta_E': 1., 'center_x': 0., 'center_y': 0.3, 'e1': 0.25, 'e2': 0.1, 'gamma': 2.05},
                     {'gamma1': 0.05, 'gamma2': -0.01}, {'theta_E': -0.3, 'center_x': 0., 'center_y': 0.04}]
        args_epl = param_class.kwargs_to_args(kwargs_in)
        npt.assert_almost_equal(args_epl[0:5], [1, 0, 0.3, 0.25, 0.1])
        kwargs_out = param_class.args_to_kwargs(args_epl)
        npt.assert_almost_equal(kwargs_out[0]['gamma'], 2.05)
        npt.assert_almost_equal(kwargs_out[1]['gamma1'] ** 2 + kwargs_out[1]['gamma2']**2, 0.12 ** 2)
        for key in kwargs_out[-1].keys():
            npt.assert_almost_equal(kwargs_out[-1][key], self.kwargs_multipole[-1][key])

        for key in kwargs_out[2].keys():
            npt.assert_almost_equal(kwargs_out[2][key], self.kwargs_multipole[2][key])

    def test_plawboxydisky_fixed_shear(self):

        param_class = PowerLawFreeShearMultipole(self.kwargs_multipole)
        npt.assert_(param_class.to_vary_index == 3)
        kwargs_in = [{'theta_E': 1., 'center_x': 0., 'center_y': 0.3, 'e1': 0.25, 'e2': 0.1, 'gamma': 2.05},
                     {'gamma1': 0.05, 'gamma2': -0.01}, {'theta_E': -0.3, 'center_x': 0., 'center_y': 0.04}]
        args_epl = param_class.kwargs_to_args(kwargs_in)
        npt.assert_almost_equal(args_epl, [1, 0, 0.3, 0.25, 0.1, 0.05, -0.01])
        kwargs_out = param_class.args_to_kwargs(args_epl)
        npt.assert_almost_equal(kwargs_out[0]['gamma'], 2.05)
        for key in kwargs_out[-1].keys():
            npt.assert_almost_equal(kwargs_out[-1][key], self.kwargs_multipole[-1][key])

        for key in kwargs_out[2].keys():
            npt.assert_almost_equal(kwargs_out[2][key], self.kwargs_multipole[2][key])

if __name__ == '__main__':
    pytest.main()