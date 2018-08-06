from lenstronomy.LensModel.Optimizer.optimizer import Optimizer
from lenstronomy.Util.util import sort_image_index
import numpy.testing as npt
import numpy as np
import pytest

class TestSinglePlaneOptimizer(object):

    x_pos_simple,y_pos_simple = np.array([ 0.69190974, -0.58959536,  0.75765166, -0.70329933]),\
                                np.array([-0.94251661,  1.01956872,  0.45230274, -0.43988017])
    magnification_simple = [1,0.9848458,0.63069122,0.54312452]

    lens_model_list_simple = ['SPEP', 'SHEAR']
    kwargs_lens_simple = [{'theta_E': 0.7, 'center_x': 0.0, 'center_y': 0, 'e1': 0.0185665252864011, 'gamma': 2.,
                           'e2': 0.08890716633399057}, {'e1': 0.00418890660015825, 'e2': -0.02908846518073248}]

    lens_model_list_subs = lens_model_list_simple + ['NFW'] * 5
    kwargs_lens_subs = kwargs_lens_simple + [{'theta_Rs': 0.005, 'center_y': -0.82, 'center_x': 0.944, 'Rs': 0.13},
                                             {'theta_Rs': 0.003, 'center_y': -0.24, 'center_x': -1.8, 'Rs': 0.23},
                                             {'theta_Rs': 0.008, 'center_y': 0.44, 'center_x': -1.8, 'Rs': 0.33},
                                             {'theta_Rs': 0.0015, 'center_y': 1.04, 'center_x': 0.8, 'Rs': 0.2},
                                             {'theta_Rs': 0.011, 'center_y': -0.4, 'center_x': 0.18, 'Rs': 0.109}]

    optimizer_simple = Optimizer(x_pos_simple, y_pos_simple, magnification_target=magnification_simple, redshift_list=[],
                                 lens_model_list=lens_model_list_simple, kwargs_lens=kwargs_lens_simple, multiplane=False, verbose=True,
                                 optimizer_routine='optimize_SPEP_shear')

    optimizer_subs = Optimizer(x_pos_simple, y_pos_simple, magnification_target=magnification_simple, redshift_list=[],
                               lens_model_list=lens_model_list_subs, kwargs_lens=kwargs_lens_subs, multiplane=False, verbose=True,
                               optimizer_routine='optimize_SPEP_shear')

    def test_single_plane_simple(self):

        kwargs_lens, source, [x_image,y_image] = self.optimizer_simple.optimize(n_particles=50, n_iterations=300,restart=2)
        index = sort_image_index(x_image, y_image, self.x_pos_simple, self.y_pos_simple)

        x_image = x_image[index]
        y_image = y_image[index]
        mags = self.optimizer_simple.optimizer_amoeba.lensModel.magnification(x_image, y_image, kwargs_lens)
        mags = np.absolute(mags)
        mags *= max(mags)**-1

        npt.assert_almost_equal(x_image, self.x_pos_simple, decimal=3)
        npt.assert_almost_equal(y_image, self.y_pos_simple, decimal=3)
        npt.assert_array_less(np.absolute(self.magnification_simple - mags)*0.15**-1,[1,1,1,1])

    def test_single_plane_subs(self,tol=0.003):

        kwargs_lens, source, [x_image,y_image] = self.optimizer_subs.optimize(n_particles=50, n_iterations=300,restart=2)
        index = sort_image_index(x_image, y_image, self.x_pos_simple, self.y_pos_simple)
        x_image = x_image[index]
        y_image = y_image[index]

        dx = np.absolute(x_image - self.x_pos_simple)
        dy = np.absolute(y_image - self.y_pos_simple)

        npt.assert_array_less(dx,[tol]*len(dx))
        npt.assert_array_less(dy,[tol]*len(dy))

if __name__ == '__main__':
    pytest.main()

