from lenstronomy.LensModel.Optimizer.optimizer import Optimizer
import numpy.testing as npt
from astropy.cosmology import FlatLambdaCDM
import pytest
from time import time
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Optimizer.fixed_routines import *
from lenstronomy.LensModel.Optimizer.single_background import SingleBackground
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions


class TestSingleBackground(object):

    def setup(self):

        np.random.seed(0)
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        kwargs_lens_simple = [{'theta_E': 0.7, 'center_x': -0.01, 'center_y': 0.001, 'e1': 0.018, 'gamma': 2.,
                               'e2': 0.089}, {'e1': 0.041, 'e2': -0.029}]
        lens_model_list_full = ['SPEP', 'SHEAR'] + ['NFW'] * 3
        z_list_full = [0.5, 0.5] + [0.4, 0.5, 0.6]
        sub_1 = {'theta_Rs': 0.007, 'center_y': -0.58, 'center_x': 0.57, 'Rs': 0.08}
        sub_2 = {'theta_Rs': 0.005, 'center_y': -0.65, 'center_x': 0.58, 'Rs': 0.08}
        sub_3 = {'theta_Rs': 0.006, 'center_y': 0.45, 'center_x': -0.6, 'Rs': 0.1}

        self.guess_lensmodel = LensModel(lens_model_list=lens_model_list_full[0:2],redshift_list=[0.5,0.5],z_source=1.5,
                                         cosmo=self.cosmo, multi_plane=True)
        self.guess_kwargs = kwargs_lens_simple

        self.kwargs_lens_full_background = kwargs_lens_simple + [sub_1, sub_2, sub_3]
        self.lensmodel_fixed_background = LensModel(lens_model_list=lens_model_list_full,
                          redshift_list=z_list_full,z_source=1.5,
                          cosmo=self.cosmo, multi_plane=True)

        self.xpos = np.array([0.57214854, -0.58080315, -0.52931625, 0.35625829])
        self.ypos = np.array([-0.59233359, 0.47546643, -0.37241599, 0.48181594])

        self.x_true, self.y_true, self.alphax_true, self.alphay_true = self.lensmodel_fixed_background.lens_model.ray_shooting_partial(
            np.zeros(4),
            np.zeros(4),
            self.xpos, self.ypos,
            z_start=0,
            z_stop=0.5,
            kwargs_lens=self.kwargs_lens_full_background[0:4])

        self.lensmodel_nobackground = LensModel(lens_model_list = lens_model_list_full[0:4], redshift_list=z_list_full[0:4],z_source=1.5,
                          cosmo=self.cosmo, multi_plane=True)
        self.kwargs_lens_nobackground = self.kwargs_lens_full_background[0:4]


    def test_delta_beta(self):

        # put no subhalos behind, assure that delta_beta is zero regardless of alpha_guess
        split = SingleBackground(self.lensmodel_nobackground, self.xpos,
                                 self.ypos, self.kwargs_lens_full_background, 1.5, 0.5,
                                 self.cosmo, [0, 1])

        delta_betax, delta_betay = split._compute_deltabeta(900, 1000)

        npt.assert_almost_equal(delta_betax,delta_betay)


        split = SingleBackground(self.lensmodel_nobackground, self.xpos,
                                 self.ypos, self.kwargs_lens_full_background, 1.5, 0.5,
                                 self.cosmo, [0, 1], guess_lensmodel=self.guess_lensmodel,
                                 guess_kwargs=self.guess_kwargs)

        _, _ = split._compute_deltabeta(900, 1000)


    def test_rayshooting(self):

        split = SingleBackground(self.lensmodel_fixed_background, self.xpos,
                                 self.ypos, self.kwargs_lens_full_background, 1.5, 0.5,
                                 self.cosmo, [0, 1])

        for index in range(0,4):

            betax, betay = split.ray_shooting(self.xpos[index],self.ypos[index],self.kwargs_lens_full_background)
            betax_fast, betay_fast = split._ray_shooting_fast(self.kwargs_lens_full_background[0:2])

            npt.assert_almost_equal(betax, betax_fast[index])
            npt.assert_almost_equal(betay, betay_fast[index])

    def test_alpha(self):

        split = SingleBackground(self.lensmodel_fixed_background, self.xpos,
                                 self.ypos, self.kwargs_lens_full_background, 1.5, 0.5,
                                 self.cosmo, [0, 1])

        ax, ay = split._alpha(self.xpos,self.ypos,self.kwargs_lens_full_background)
        axfast, ayfast = split._alpha_fast(self.kwargs_lens_full_background[0:2],0)
        npt.assert_almost_equal(ay, ayfast)
        npt.assert_almost_equal(ay, ayfast)

        ax, ay = split._alpha(self.xpos+0.00000001, self.ypos, self.kwargs_lens_full_background)
        axfast, ayfast = split._alpha_fast(self.kwargs_lens_full_background[0:2], 1)
        npt.assert_almost_equal(ay, ayfast)
        npt.assert_almost_equal(ay, ayfast)

        ax, ay = split._alpha(self.xpos, self.ypos + 0.00000001, self.kwargs_lens_full_background)
        axfast, ayfast = split._alpha_fast(self.kwargs_lens_full_background[0:2], 2)
        npt.assert_almost_equal(ay, ayfast)
        npt.assert_almost_equal(ay, ayfast)

    def test_hessian(self):

        split = SingleBackground(self.lensmodel_fixed_background, self.xpos,
                                 self.ypos, self.kwargs_lens_full_background, 1.5, 0.5,
                                 self.cosmo, [0, 1])

        out1 = split.hessian(self.xpos, self.ypos, self.kwargs_lens_full_background)

        out2 = split._hessian_fast(self.kwargs_lens_full_background[0:2])

        for (val1,val2) in zip(out1,out2):
            npt.assert_almost_equal(val1,val2)


    def test_magnification(self):

        split = SingleBackground(self.lensmodel_fixed_background, self.xpos,
                                 self.ypos, self.kwargs_lens_full_background, 1.5, 0.5,
                                 self.cosmo, [0, 1])
        out1 = split.magnification(self.xpos,self.ypos,self.kwargs_lens_full_background)
        out2 = split._magnification_fast(self.kwargs_lens_full_background[0:2])
        for (val1,val2) in zip(np.absolute(out1),out2):
            npt.assert_almost_equal(val1,val2)

    def test_mag_finite(self):

        split = SingleBackground(self.lensmodel_fixed_background, self.xpos,
                                 self.ypos, self.kwargs_lens_full_background, 1.5, 0.5,
                                 self.cosmo, [0, 1])


        mag_point = np.absolute(split.magnification(self.xpos, self.ypos,self.kwargs_lens_full_background))
        mag_point_fast = np.absolute(split.magnification(self.xpos, self.ypos,self.kwargs_lens_full_background))
        extension = LensModelExtensions(split)

        mag_finite = extension.magnification_finite(self.xpos, self.ypos,self.kwargs_lens_full_background,source_sigma=0.00001,
                                             window_size=0.01,grid_number=2000)
        for (mp, mpf, mfin) in zip(mag_point,mag_point_fast, mag_finite):
            npt.assert_almost_equal(mp,mpf,3)
            npt.assert_almost_equal(mpf,mfin,3)

if __name__ == '__main__':
    pytest.main()