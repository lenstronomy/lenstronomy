__author__ = 'sibirrer'

import numpy.testing as npt
import pytest
import numpy as np

from lenstronomy.Analysis.td_cosmography import TDCosmography
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver


class TestTDCosmography(object):

    def setup(self):
        kwargs_model = {'lens_light_model_list': ['HERNQUIST'],
                        'lens_model_list': ['SIE'],
                        'point_source_model_list': ['LENSED_POSITION']}
        z_lens = 0.5
        z_source = 2.5
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        self.td_cosmo = TDCosmography(z_lens, z_source, kwargs_model, cosmo_fiducial=cosmo, lens_model_kinematics_bool=None,
                 light_model_kinematics_bool=None)
        self.lens = LensModel(lens_model_list=['SIE'], cosmo=cosmo, z_lens=z_lens, z_source=z_source)
        self.solver = LensEquationSolver(lensModel=self.lens)

        self.kwargs_lens = [{'theta_E': 1, 'e1': 0.1, 'e2': -0.2, 'center_x': 0, 'center_y': 0}]
        source_x, source_y = 0, 0.05
        image_x, image_y = self.solver.image_position_from_source(source_x, source_y, self.kwargs_lens, min_distance=0.1, search_window=10)
        self.kwargs_ps = [{'ra_image': image_x, 'dec_image': image_y}]
        self.image_x, self.image_y = image_x, image_y

    def test_time_delays(self):
        dt = self.td_cosmo.time_delays(self.kwargs_lens, self.kwargs_ps, kappa_ext=0)
        dt_true = self.lens.arrival_time(self.image_x, self.image_y, self.kwargs_lens)
        npt.assert_almost_equal(dt, dt_true, decimal=6)

    def test_fermat_potential(self):
        fermat_pot = self.td_cosmo.fermat_potential(self.kwargs_lens, self.kwargs_ps)
        fermat_pot_true = self.lens.fermat_potential(self.image_x, self.image_y, self.kwargs_lens)
        npt.assert_almost_equal(fermat_pot, fermat_pot_true, decimal=6)

        diff = 0.1
        kwargs_ps = [{'ra_image': self.image_x+diff, 'dec_image': self.image_y}]
        fermat_pot = self.td_cosmo.fermat_potential(self.kwargs_lens, kwargs_ps)
        fermat_pot_true = self.lens.fermat_potential(self.image_x+diff, self.image_y, self.kwargs_lens)
        ratio = fermat_pot / fermat_pot_true
        assert np.max(np.abs(ratio)) > 1.05

"""

    def test_time_delays(self):
        z_lens = 0.5
        z_source = 1.5
        kwargs_options = {'lens_model_list': ['SPEP'], 'point_source_model_list': ['LENSED_POSITION']}
        e1, e2 = param_util.phi_q2_ellipticity(0, 0.7)
        kwargs_lens = [{'theta_E': 1, 'gamma': 2, 'e1': e1, 'e2': e2}]
        kwargs_else = [{'ra_image': [-1, 0, 1], 'dec_image': [0, 0, 0]}]
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        lensProp = KinematicAPI(z_lens, z_source, kwargs_options, cosmo=cosmo)
        delays = lensProp.time_delays(kwargs_lens, kwargs_ps=kwargs_else, kappa_ext=0)
        npt.assert_almost_equal(delays[0], -31.387590264501007, decimal=8)
        npt.assert_almost_equal(delays[1], 0, decimal=8)
        npt.assert_almost_equal(delays[2], -31.387590264501007, decimal=8)
        kappa_ext = 0.1
        delays_kappa = lensProp.time_delays(kwargs_lens, kwargs_ps=kwargs_else, kappa_ext=kappa_ext)
        npt.assert_almost_equal(delays_kappa/(1.-kappa_ext), delays, decimal=8)

        kappa_ext = 0.1
        delays_kappa = lensProp.time_delays(kwargs_lens, kwargs_ps=kwargs_else, kappa_ext=kappa_ext)
        npt.assert_almost_equal(delays_kappa / (1. - kappa_ext), delays, decimal=8)

"""

if __name__ == '__main__':
    pytest.main()
