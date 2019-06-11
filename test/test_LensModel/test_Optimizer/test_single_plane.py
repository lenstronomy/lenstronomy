from lenstronomy.LensModel.Optimizer.optimizer import Optimizer
from lenstronomy.Util.util import sort_image_index
import numpy.testing as npt
import numpy as np
import pytest


class TestSinglePlaneOptimizer(object):

    np.random.seed(0)
    x_pos_simple,y_pos_simple = np.array([ 0.69190974, -0.58959536,  0.75765166, -0.70329933]),\
                                np.array([-0.94251661,  1.01956872,  0.45230274, -0.43988017])
    magnification_simple = [1., 0.9848458, 0.63069122, 0.54312452]

    lens_model_list_simple = ['SPEP', 'SHEAR']
    kwargs_lens_simple = [{'theta_E': 0.7, 'center_x': 0.0, 'center_y': 0, 'e1': 0.0185665252864011, 'gamma': 2.,
                           'e2': 0.08890716633399057}, {'e1': 0.00418890660015825, 'e2': -0.02908846518073248}]

    lens_model_list_subs = lens_model_list_simple + ['NFW'] * 5
    kwargs_lens_subs = kwargs_lens_simple + [{'alpha_Rs': 0.005, 'center_y': -0.82, 'center_x': 0.944, 'Rs': 0.13},
                                             {'alpha_Rs': 0.003, 'center_y': -0.24, 'center_x': -1.8, 'Rs': 0.23},
                                             {'alpha_Rs': 0.008, 'center_y': 0.44, 'center_x': -1.8, 'Rs': 0.33},
                                             {'alpha_Rs': 0.0015, 'center_y': 1.04, 'center_x': 0.8, 'Rs': 0.2},
                                             {'alpha_Rs': 0.011, 'center_y': -0.4, 'center_x': 0.18, 'Rs': 0.109}]

    kwargs_lens_subs = kwargs_lens_subs
    optimizer_simple = Optimizer(x_pos_simple, y_pos_simple, magnification_target=magnification_simple, redshift_list=[],
                                 lens_model_list=lens_model_list_simple, kwargs_lens=kwargs_lens_simple, multiplane=False, verbose=True,
                                 optimizer_routine='fixed_powerlaw_shear')

    optimizer_subs = Optimizer(x_pos_simple, y_pos_simple, magnification_target=magnification_simple, redshift_list=[],
                               lens_model_list=lens_model_list_subs, kwargs_lens=kwargs_lens_subs, multiplane=False, verbose=True,
                               optimizer_routine='fixed_powerlaw_shear')

    optimizer_image_plane = Optimizer(x_pos_simple, y_pos_simple, magnification_target=magnification_simple, redshift_list=[],
                                 lens_model_list=lens_model_list_simple, kwargs_lens=kwargs_lens_simple, multiplane=False, verbose=True,
                                 optimizer_routine='fixed_powerlaw_shear', chi2_mode='image', tol_image=0.006, pso_convergence_mean=100)

    def test_single_plane_simple(self):

        kwargs_lens, source, [x_image,y_image] = self.optimizer_simple.optimize(n_particles=30, n_iterations=30,restart=2)

        mags = self.optimizer_simple._lensModel.magnification(x_image, y_image, kwargs_lens)

    def test_single_plane_subs(self,tol=0.003):

        kwargs_lens, source, [x_image,y_image] = self.optimizer_subs.optimize(n_particles=30, n_iterations=30,restart=2)
        mags = self.optimizer_subs._lensModel.magnification(x_image, y_image, kwargs_lens)

    def test_image_plane_chi2(self):
        kwargs_lens, source, [x_image, y_image] = self.optimizer_image_plane.optimize(n_particles=20, n_iterations=150, restart=1)


if __name__ == '__main__':
    pytest.main()
