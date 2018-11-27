__author__ = 'dgilman'


from lenstronomy.LensModel.Profiles.coreBurkert import coreBurkert
from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.LensModel.Profiles.hybrid import HYBRID

import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
import pytest

class TestHYBRID(object):
    """
    tests the Gaussian methods
    """

    def _interpolating_function(self, kwargs1, kwargs2):

        rs = kwargs1['Rs']
        r_core = kwargs2['r_core']
        power = 4
        coeff = 1
        q_crit = 0.3
        ratio = (r_core * rs**-1) * q_crit**-1
        arg = coeff*(ratio)**power
        f = np.exp(-arg)

        return f

    def setup(self):

        lens_model_2 = coreBurkert()
        lens_model_1 = NFW()

        self.nfw = lens_model_1
        self.cb = lens_model_2

        self.hybrid = HYBRID('NFW', 'coreBURKERT', self._interpolating_function)

    def test_derivatives(self):

        theta_Rs = 1
        Rs = 5
        q = 0.5
        r_core = q*Rs

        R = np.linspace(0.25*Rs, 5*Rs, 1000)
        kwargs1 = {'x': R, 'y':0, 'Rs': Rs, 'theta_Rs': theta_Rs}
        kwargs2 = {'x': R, 'y':0, 'Rs': Rs, 'theta_Rs': theta_Rs, 'r_core': r_core}

        dxn = self.nfw.derivatives(**kwargs1)[0]
        dxcb = self.cb.derivatives(**kwargs2)[0]
        dxint = self.hybrid.derivatives(kwargs1, kwargs2)[0]
        f = self._interpolating_function(kwargs1, kwargs2)
        npt.assert_almost_equal(dxn*f + dxcb*(1-f), dxint)

        theta_Rs = 1
        Rs = 5
        q = 0.01
        r_core = q * Rs

        R = np.linspace(0.25 * Rs, 5 * Rs, 1000)
        kwargs1 = {'x': R, 'y': 0, 'Rs': Rs, 'theta_Rs': theta_Rs}
        kwargs2 = {'x': R, 'y': 0, 'Rs': Rs, 'theta_Rs': theta_Rs, 'r_core': r_core}

        dxn = self.nfw.derivatives(**kwargs1)[0]
        dxint = self.hybrid.derivatives(kwargs1, kwargs2)[0]
        f = self._interpolating_function(kwargs1, kwargs2)
        npt.assert_almost_equal(dxn, dxint)

    def test_hessian(self):

        theta_Rs = 1
        Rs = 5
        q = 0.5
        r_core = q*Rs

        R = np.linspace(0.25*Rs, 5*Rs, 1000)
        kwargs1 = {'x': R, 'y':0, 'Rs': Rs, 'theta_Rs': theta_Rs}
        kwargs2 = {'x': R, 'y':0, 'Rs': Rs, 'theta_Rs': theta_Rs, 'r_core': r_core}

        dxn = self.nfw.hessian(**kwargs1)[0]
        dxcb = self.cb.hessian(**kwargs2)[0]
        dxint = self.hybrid.hessian(kwargs1, kwargs2)[0]
        f = self._interpolating_function(kwargs1, kwargs2)
        npt.assert_almost_equal(dxn*f + dxcb*(1-f), dxint)

        theta_Rs = 1
        Rs = 5
        q = 0.01
        r_core = q * Rs

        R = np.linspace(0.25 * Rs, 5 * Rs, 1000)
        kwargs1 = {'x': R, 'y': 0, 'Rs': Rs, 'theta_Rs': theta_Rs}
        kwargs2 = {'x': R, 'y': 0, 'Rs': Rs, 'theta_Rs': theta_Rs, 'r_core': r_core}

        dxn = self.nfw.hessian(**kwargs1)[0]
        dxint = self.hybrid.hessian(kwargs1, kwargs2)[0]
        npt.assert_almost_equal(dxn, dxint)

    def test_mass(self):

        theta_Rs = 1
        Rs = 5
        q = 0.5
        r_core = q*Rs

        R = np.linspace(0.25*Rs, 5*Rs, 1000)
        kwargs1 = {'R': R, 'Rs': Rs, 'rho0': theta_Rs}
        kwargs2 = {'R': R, 'Rs': Rs, 'rho0': theta_Rs, 'r_core': r_core}

        dxn = self.nfw.mass_2d(**kwargs1)
        dxcb = self.cb.mass_2d(**kwargs2)
        dxint = self.hybrid.mass_2d(kwargs1, kwargs2)
        f = self._interpolating_function(kwargs1, kwargs2)
        npt.assert_almost_equal(dxn*f + dxcb*(1-f), dxint)

        rho = 1
        Rs = 5
        q = 0.001
        r_core = q * Rs

        R = np.linspace(0.25 * Rs, 5 * Rs, 1000)
        kwargs1 = {'R': R, 'Rs': Rs, 'rho0': theta_Rs}
        kwargs2 = {'R': R, 'Rs': Rs, 'rho0': theta_Rs, 'r_core': r_core}

        dxn = self.nfw.mass_2d(**kwargs1)
        dxint = self.hybrid.mass_2d(kwargs1, kwargs2)
        npt.assert_almost_equal(dxn, dxint, decimal=6)

        theta_Rs = 1
        Rs = 5
        q = 0.5
        r_core = q * Rs

        R = np.linspace(0.25 * Rs, 5 * Rs, 1000)
        kwargs1 = {'R': R, 'Rs': Rs, 'rho0': theta_Rs}
        kwargs2 = {'R': R, 'Rs': Rs, 'rho0': theta_Rs, 'r_core': r_core}

        dxn = self.nfw.mass_3d(**kwargs1)
        dxcb = self.cb.mass_3d(**kwargs2)
        dxint = self.hybrid.mass_3d(kwargs1, kwargs2)
        f = self._interpolating_function(kwargs1, kwargs2)
        npt.assert_almost_equal(dxn * f + dxcb * (1 - f), dxint)

        theta_Rs = 1
        Rs = 5
        q = 0.001
        r_core = q * Rs

        R = np.linspace(0.25 * Rs, 5 * Rs, 1000)
        kwargs1 = {'R': R, 'Rs': Rs, 'rho0': theta_Rs}
        kwargs2 = {'R': R, 'Rs': Rs, 'rho0': theta_Rs, 'r_core': r_core}

        dxn = self.nfw.mass_3d(**kwargs1)
        dxint = self.hybrid.mass_3d(kwargs1, kwargs2)
        npt.assert_almost_equal(dxn, dxint, decimal=6)

        theta_Rs = 1
        Rs = 5
        q = 0.5
        r_core = q * Rs

        R = np.linspace(0.25 * Rs, 5 * Rs, 1000)
        kwargs1 = {'x': R, 'y':0, 'Rs': Rs ,'rho0': theta_Rs}
        kwargs2 = {'x': R, 'y':0, 'Rs': Rs,'rho0': theta_Rs, 'r_core': r_core}

        dxn = self.nfw.density_2d(**kwargs1)
        dxcb = self.cb.density_2d(**kwargs2)
        dxint = self.hybrid.density_2d(kwargs1, kwargs2)
        f = self._interpolating_function(kwargs1, kwargs2)
        npt.assert_almost_equal(dxn * f + dxcb * (1 - f), dxint)

        theta_Rs = 1
        Rs = 5
        q = 0.001
        r_core = q * Rs

        R = np.linspace(0.25 * Rs, 5 * Rs, 1000)
        kwargs1 = {'x': R, 'y':0, 'Rs': Rs, 'rho0': theta_Rs}
        kwargs2 = {'x': R, 'y':0, 'Rs': Rs, 'rho0': theta_Rs, 'r_core': r_core}

        dxn = self.nfw.density_2d(**kwargs1)
        dxint = self.hybrid.density_2d(kwargs1, kwargs2)
        npt.assert_almost_equal(dxn, dxint, decimal=6)

        theta_Rs = 1
        Rs = 5
        q = 0.5
        r_core = q * Rs

        R = np.linspace(0.25 * Rs, 5 * Rs, 1000)
        kwargs1 = {'R': R, 'Rs': Rs, 'rho0': theta_Rs}
        kwargs2 = {'R': R, 'Rs': Rs, 'rho0': theta_Rs, 'r_core': r_core}

        dxn = self.nfw.density(**kwargs1)
        dxcb = self.cb.density(**kwargs2)
        dxint = self.hybrid.density(kwargs1, kwargs2)
        f = self._interpolating_function(kwargs1, kwargs2)
        npt.assert_almost_equal(dxn * f + dxcb * (1 - f), dxint)

        theta_Rs = 1
        Rs = 5
        q = 0.001
        r_core = q * Rs

        R = np.linspace(0.25 * Rs, 5 * Rs, 1000)
        kwargs1 = {'R': R, 'Rs': Rs, 'rho0': theta_Rs}
        kwargs2 = {'R': R, 'Rs': Rs, 'rho0': theta_Rs, 'r_core': r_core}

        dxn = self.nfw.density(**kwargs1)
        dxint = self.hybrid.density(kwargs1, kwargs2)
        npt.assert_almost_equal(dxn, dxint, decimal=6)

    def test_func(self):
        theta_Rs = 1
        Rs = 5
        q = 0.5
        r_core = q * Rs

        R = np.linspace(0.25 * Rs, 5 * Rs, 1000)
        kwargs1 = {'x': R, 'y': 0, 'Rs': Rs, 'theta_Rs': theta_Rs}
        kwargs2 = {'x': R, 'y': 0, 'Rs': Rs, 'theta_Rs': theta_Rs, 'r_core': r_core}

        dxn = self.nfw.function(**kwargs1)
        dxcb = self.cb.function(**kwargs2)
        dxint = self.hybrid.function(kwargs1, kwargs2)
        f = self._interpolating_function(kwargs1, kwargs2)
        npt.assert_almost_equal(dxn * f + dxcb * (1 - f), dxint)

        theta_Rs = 1
        Rs = 5
        q = 0.001
        r_core = q * Rs

        R = np.linspace(0.25 * Rs, 5 * Rs, 1000)
        kwargs1 = {'x': R, 'y': 0, 'Rs': Rs, 'theta_Rs': theta_Rs}
        kwargs2 = {'x': R, 'y': 0, 'Rs': Rs, 'theta_Rs': theta_Rs, 'r_core': r_core}

        dxn = self.nfw.function(**kwargs1)
        dxint = self.hybrid.function(kwargs1, kwargs2)
        npt.assert_almost_equal(dxn, dxint)

    def test_LensModel(self):

        rho0 = 1
        Rs = 5
        q = 0.5
        r_core = q * Rs

        R = np.linspace(0.25 * Rs, 5 * Rs, 5)
        kwargs1 = {'x': R, 'y':0, 'Rs': Rs, 'theta_Rs': rho0}
        kwargs2 = {'x': R, 'y':0, 'Rs': Rs, 'theta_Rs': rho0, 'r_core': r_core}
        kwargs_lens1 = {'Rs': Rs, 'theta_Rs': rho0}
        kwargs_lens2 = {'Rs': Rs, 'theta_Rs': rho0, 'r_core': r_core}

        kwargs = {'kwargs1': kwargs_lens1, 'kwargs2': kwargs_lens2}

        from lenstronomy.LensModel.lens_model import LensModel
        lens_model_list = ['HYBRID']
        redshift_list = [0.5]
        kwargs_lensmodel = {'lens_model_1': 'NFW', 'lens_model_2': 'coreBURKERT',
                            'interpolating_function': self._interpolating_function}

        lensmodel_single = LensModel(lens_model_list, z_lens=0.5, z_source=1.5,
                               redshift_list=redshift_list, multi_plane=False,
                               kwargs_lensmodel=kwargs_lensmodel)

        #deflection_interpolated = self.hybrid.derivatives(kwargs1, kwargs2)[0]
        density_interpolated_single = lensmodel_single.alpha(R, 0, [kwargs])[0]

        #npt.assert_almost_equal(density_interpolated_single, deflection_interpolated)

t = TestHYBRID()
t.setup()
t.test_LensModel()
exit(1)
if __name__ == '__main__':
    pytest.main()