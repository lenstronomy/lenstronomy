__author__ = 'dgilman'


from lenstronomy.LensModel.Profiles.coreBurkert import coreBurkert
from lenstronomy.LensModel.Profiles.tnfw import TNFW
from lenstronomy.LensModel.Profiles.hybrid import HYBRID
from lenstronomy.LensModel.lens_model import LensModel

import numpy as np
import numpy.testing as npt
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
        lens_model_1 = TNFW()

        self.nfw = lens_model_1
        self.cb = lens_model_2

        self.hybrid = HYBRID('TNFW', 'coreBURKERT', self._interpolating_function)

    def test_lensing(self):

        theta_Rs = 1
        Rs = 5
        q = 0.5
        r_core = q*Rs
        x = np.linspace(0.2*Rs, 5*Rs, 20)
        y = 0
        kwargs1 = {'Rs': Rs, 'theta_Rs': theta_Rs, 'r_trunc': 10*Rs}
        kwargs2 = {'Rs': Rs, 'theta_Rs': theta_Rs, 'r_core': r_core}

        dxn = self.nfw.function(x, y, **kwargs1)
        dxcb = self.cb.function(x, y, **kwargs2)
        dxint = self.hybrid.function(x, y, kwargs1, kwargs2)
        f = self._interpolating_function(kwargs1, kwargs2)
        npt.assert_almost_equal(dxn * f + dxcb * (1 - f), dxint)

        dxn = self.nfw.derivatives(x, y, **kwargs1)[0]
        dxcb = self.cb.derivatives(x, y, **kwargs2)[0]
        dxint = self.hybrid.derivatives(x, y, kwargs1, kwargs2)[0]
        f = self._interpolating_function(kwargs1, kwargs2)
        npt.assert_almost_equal(dxn*f + dxcb*(1-f), dxint)

        dxn = self.nfw.hessian(x, y, **kwargs1)[0]
        dxcb = self.cb.hessian(x, y, **kwargs2)[0]
        dxint = self.hybrid.hessian(x, y, kwargs1, kwargs2)[0]
        f = self._interpolating_function(kwargs1, kwargs2)
        npt.assert_almost_equal(dxn * f + dxcb * (1 - f), dxint)

        q = 0.0001
        r_core = q * Rs
        kwargs2['r_core'] = r_core

        dxn = self.nfw.function(x, y, **kwargs1)
        dxint = self.hybrid.function(x, y, kwargs1, kwargs2)
        npt.assert_almost_equal(dxn, dxint)

        dxn = self.nfw.derivatives(x, y, **kwargs1)[0]
        dxint = self.hybrid.derivatives(x, y, kwargs1, kwargs2)[0]
        npt.assert_almost_equal(dxn, dxint)

        dxn = self.nfw.hessian(x, y, **kwargs1)[0]
        dxint = self.hybrid.hessian(x, y, kwargs1, kwargs2)[0]
        npt.assert_almost_equal(dxn, dxint)

    def test_mass(self):

        theta_Rs = 1
        Rs = 5
        q = 0.5
        r_core = q*Rs

        R = np.linspace(0.25*Rs, 5*Rs, 1000)
        kwargs1 = {'Rs': Rs, 'rho0': theta_Rs, 'r_trunc': 10*Rs}
        kwargs2 = {'Rs': Rs, 'rho0': theta_Rs, 'r_core': r_core}

        dxn = self.nfw.mass_2d(R, **kwargs1)
        dxcb = self.cb.mass_2d(R, **kwargs2)
        dxint = self.hybrid.mass_2d(R, kwargs1, kwargs2)
        f = self._interpolating_function(kwargs1, kwargs2)
        npt.assert_almost_equal(dxn*f + dxcb*(1-f), dxint)

        Rs = 5
        q = 0.001
        r_core = q * Rs
        kwargs1 = {'Rs': Rs, 'rho0': theta_Rs, 'r_trunc': 10 * Rs}
        kwargs2 = {'Rs': Rs, 'rho0': theta_Rs, 'r_core': r_core}

        R = np.linspace(0.25 * Rs, 5 * Rs, 1000)

        dxn = self.nfw.mass_2d(R, **kwargs1)
        dxint = self.hybrid.mass_2d(R, kwargs1, kwargs2)
        npt.assert_almost_equal(dxn, dxint, decimal=6)


        theta_Rs = 1
        Rs = 5
        q = 0.5
        r_core = q * Rs

        R = np.linspace(0.25 * Rs, 5 * Rs, 1000)
        kwargs1 = {'Rs': Rs, 'rho0': theta_Rs, 'r_trunc': 10 * Rs}
        kwargs2 = {'Rs': Rs, 'rho0': theta_Rs, 'r_core': r_core}

        dxn = self.nfw.density_2d(R, 0, **kwargs1)
        dxcb = self.cb.density_2d(R, 0,  **kwargs2)
        dxint = self.hybrid.density_2d(R,0, kwargs1, kwargs2)
        f = self._interpolating_function(kwargs1, kwargs2)
        npt.assert_almost_equal(dxn * f + dxcb * (1 - f), dxint)

        theta_Rs = 1
        Rs = 5
        q = 0.001
        r_core = q * Rs

        R = np.linspace(0.25 * Rs, 5 * Rs, 1000)
        kwargs1 = {'Rs': Rs, 'rho0': theta_Rs, 'r_trunc': 10 * Rs}
        kwargs2 = {'Rs': Rs, 'rho0': theta_Rs, 'r_core': r_core}

        dxn = self.nfw.density_2d(R,0, **kwargs1)
        dxint = self.hybrid.density_2d(R,0, kwargs1, kwargs2)
        npt.assert_almost_equal(dxn, dxint, decimal=6)

        theta_Rs = 1
        Rs = 5
        q = 0.5
        r_core = q * Rs

        R = np.linspace(0.25 * Rs, 5 * Rs, 1000)
        kwargs1 = {'Rs': Rs, 'rho0': theta_Rs, 'r_trunc': 5*Rs}
        kwargs2 = {'Rs': Rs, 'rho0': theta_Rs, 'r_core': r_core}

        dxn = self.nfw.density(R, **kwargs1)
        dxcb = self.cb.density(R, **kwargs2)
        dxint = self.hybrid.density(R, kwargs1, kwargs2)
        f = self._interpolating_function(kwargs1, kwargs2)
        npt.assert_almost_equal(dxn * f + dxcb * (1 - f), dxint)

        theta_Rs = 1
        Rs = 5
        q = 0.001
        r_core = q * Rs

        R = np.linspace(0.25 * Rs, 5 * Rs, 1000)
        kwargs1 = {'Rs': Rs, 'rho0': theta_Rs, 'r_trunc': 5*Rs}
        kwargs2 = {'Rs': Rs, 'rho0': theta_Rs, 'r_core': r_core}

        dxn = self.nfw.density(R, **kwargs1)
        dxint = self.hybrid.density(R, kwargs1, kwargs2)
        npt.assert_almost_equal(dxn, dxint, decimal=6)

    def test_LensModel(self):

        rho0 = 1
        Rs = 5
        q = 0.5
        r_core = q * Rs
        x = np.linspace(0.2*Rs, 5*Rs, 20)
        y = 0

        kwargs1 = {'Rs': Rs, 'theta_Rs': rho0, 'r_trunc': 5*Rs}
        kwargs2 = {'Rs': Rs, 'theta_Rs': rho0, 'r_core': r_core}

        kwargs = {'kwargs1': kwargs1, 'kwargs2': kwargs2}

        lens_model_list = ['HYBRID']
        kwargs_lensmodel = {'lens_model_1': 'TNFW', 'lens_model_2': 'coreBURKERT',
                            'interpolating_function': self._interpolating_function}

        lensmodel_single = LensModel(lens_model_list, kwargs_lensmodel=kwargs_lensmodel)
        lensmodel_multi = LensModel(lens_model_list, redshift_list=[0.5], multi_plane=True,
                                    kwargs_lensmodel=kwargs_lensmodel, z_source=1.5)

        deflection_interpolated = self.hybrid.derivatives(x, y, kwargs1, kwargs2)[0]
        deflection_single = lensmodel_single.alpha(x, 0, [kwargs])[0]
        deflection_multi = lensmodel_multi.alpha(x, 0 ,[kwargs])[0]

        npt.assert_almost_equal(deflection_interpolated, deflection_single)
        npt.assert_almost_equal(deflection_single, deflection_multi)

if __name__ == '__main__':
    pytest.main()