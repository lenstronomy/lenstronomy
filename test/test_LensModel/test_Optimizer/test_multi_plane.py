from lenstronomy.LensModel.Optimizer.optimizer import Optimizer
import numpy.testing as npt
from astropy.cosmology import FlatLambdaCDM
import pytest
from time import time
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Optimizer.fixed_routines import *
from lenstronomy.LensModel.Optimizer.multi_plane import MultiPlaneLensing
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions


class TestMultiPlaneOptimizer(object):

    def setup(self):

        np.random.seed(0)
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        self.x_pos_simple = np.array([-0.45328229, 0.57461556, 0.53757501, -0.42312438])
        self.y_pos_simple = np.array([0.69582971, -0.51226356, 0.37577509, -0.40245467])
        self.magnification_simple = np.array([2.79394452, 3.28101725, 2.29495699, 1.63409843]) * 3.28101725 ** -1

        redshift_list_simple = [0.5, 0.5]
        lens_model_list_simple = ['SPEP', 'SHEAR']
        self.kwargs_lens_simple = [
            {'theta_E': 0.7, 'center_x': 0.0, 'center_y': 0, 'e1': 0.0185665252864011, 'gamma': 2.,
             'e2': 0.08890716633399057}, {'e1': 0.00418890660015825, 'e2': -0.02908846518073248}]

        front_halos = ['NFW', 'NFW', 'NFW', 'NFW', 'NFW', 'NFW']
        front_redshifts = [0.4, 0.4, 0.4, 0.44, 0.44, 0.44]
        self.front_args = [{'theta_Rs': 0.001, 'center_y': 0.2, 'center_x': 1.2, 'Rs': 0.13},
                           {'theta_Rs': 0.002, 'center_y': -0.2, 'center_x': 1, 'Rs': 0.11},
                           {'theta_Rs': 0.004, 'center_y': 0.12, 'center_x': -1.2, 'Rs': 0.13},
                           {'theta_Rs': 0.0001, 'center_y': 0.32, 'center_x': -0.2, 'Rs': 0.13},
                           {'theta_Rs': 0.003, 'center_y': 0.82, 'center_x': 0.78, 'Rs': 0.13},
                           {'theta_Rs': 0.008, 'center_y': 1, 'center_x': 0.75, 'Rs': 0.16}]

        main_halos = ['NFW', 'NFW', 'NFW', 'NFW', 'NFW', 'NFW']
        main_redshifts = [0.5] * 6
        self.main_args = [{'theta_Rs': 0.001, 'center_y': 1.2, 'center_x': 0.2, 'Rs': 0.13},
                          {'theta_Rs': 0.002, 'center_y': -0.1, 'center_x': 0.91, 'Rs': 0.11},
                          {'theta_Rs': 0.009, 'center_y': 0.18, 'center_x': -0.42, 'Rs': 0.13},
                          {'theta_Rs': 0.0001, 'center_y': 0.42, 'center_x': -0.92, 'Rs': 0.13},
                          {'theta_Rs': 0.005, 'center_y': 0.9, 'center_x': 0.48, 'Rs': 0.13},
                          {'theta_Rs': 0.008, 'center_y': -1, 'center_x': 0.95, 'Rs': 0.16}]


        back_halos = ['NFW', 'NFW', 'NFW', 'NFW', 'NFW', 'NFW']
        back_redshifts = [0.55, 0.6, 0.6, 0.74, 0.74, 0.8]
        self.back_args = [{'theta_Rs': 0.004, 'center_y': 0.1, 'center_x': 1, 'Rs': 0.13},
                          {'theta_Rs': 0.001, 'center_y': 0.2, 'center_x': 0.7, 'Rs': 0.11},
                          {'theta_Rs': 0.003, 'center_y': -0.1, 'center_x': -1, 'Rs': 0.13},
                          {'theta_Rs': 0.0008, 'center_y': 0.42, 'center_x': 0.1, 'Rs': 0.13},
                          {'theta_Rs': 0.0014, 'center_y': 0.42, 'center_x': 1.08, 'Rs': 0.13},
                          {'theta_Rs': 0.006, 'center_y': 0.5, 'center_x': 0.75, 'Rs': 0.16}]

        lens_model_list_full = lens_model_list_simple + front_halos + main_halos + back_halos
        redshift_list_full = redshift_list_simple + front_redshifts + main_redshifts + back_redshifts
        self.kwargs_lens_full = self.kwargs_lens_simple + self.front_args + self.main_args + self.back_args

        self.lens_model_full = LensModel(lens_model_list_full, z_source=1.5, redshift_list=redshift_list_full,
                                         cosmo=self.cosmo,
                                         multi_plane=True)

        self.lens_model_front = LensModel(front_halos + main_halos, redshift_list=front_redshifts + main_redshifts,
                                          z_source=1.5,
                                          cosmo=self.cosmo, multi_plane=True)
        self.kwargs_front = self.front_args + self.main_args

        self.lens_model_simple = LensModel(lens_model_list_simple, z_source=1.5, redshift_list=redshift_list_simple,
                                         cosmo=self.cosmo,
                                         multi_plane=True)

        self.optimizer_simple = Optimizer(self.x_pos_simple, self.y_pos_simple,
                                          magnification_target=self.magnification_simple,
                                          redshift_list=redshift_list_simple,
                                          lens_model_list=lens_model_list_simple, kwargs_lens=self.kwargs_lens_simple,
                                          multiplane=True, verbose=True, z_source=1.5, z_main=0.5,
                                          astropy_instance=self.cosmo, optimizer_routine='fixed_powerlaw_shear')

        self.optimizer_subs = Optimizer(self.x_pos_simple, self.y_pos_simple,
                                        magnification_target=self.magnification_simple,
                                        redshift_list=redshift_list_full,
                                        lens_model_list=lens_model_list_full, kwargs_lens=self.kwargs_lens_full,
                                        multiplane=True, verbose=True, z_source=1.5, z_main=0.5,
                                        astropy_instance=self.cosmo,optimizer_routine='fixed_powerlaw_shear')

        self.optimizer_single_background = Optimizer(self.x_pos_simple, self.y_pos_simple,
                                        magnification_target=self.magnification_simple,
                                        redshift_list=redshift_list_full,
                                        lens_model_list=lens_model_list_full, kwargs_lens=self.kwargs_lens_full,
                                        multiplane=True, verbose=True, z_source=1.5, z_main=0.5,
                                        astropy_instance=self.cosmo, optimizer_routine='fixed_powerlaw_shear',single_background=True)

        self.optimizer_params = Optimizer(self.x_pos_simple, self.y_pos_simple,
                                        magnification_target=self.magnification_simple,
                                        redshift_list=redshift_list_full,
                                        lens_model_list=lens_model_list_full, kwargs_lens=self.kwargs_lens_full,
                                        multiplane=True, verbose=True, z_source=1.5, z_main=0.5,
                                        astropy_instance=self.cosmo,optimizer_routine='fixed_powerlaw_shear',
                                        constrain_params={'shear':[0.06,0.01],'shear_pa':[-30,10],'theta_E':[1,0.1]})

        lens_model_list_simple_background = ['SPEP', 'SHEAR']
        z_list_simple = [0.5, 0.5]
        kwargs_lens_simple_background = [{'theta_E': 0.7, 'center_x': -0.01, 'center_y': 0.001, 'e1': 0.018, 'gamma': 2.,
                               'e2': 0.089}, {'e1': 0.0041, 'e2': -0.029}]
        sub_1 = {'theta_Rs': 0.01, 'center_y': 0.55, 'center_x': -0.54, 'Rs': 0.13}
        sub_2 = {'theta_Rs': 0.012, 'center_y': -0.3, 'center_x': -0.1, 'Rs': 0.1}
        sub_3 = {'theta_Rs': 0.0011, 'center_y': 0.92, 'center_x': -0.81, 'Rs': 0.13}
        sub_4 = {'theta_Rs': 0.0009, 'center_y': 0.4, 'center_x': -0.41, 'Rs': 0.13}
        sub_5 = {'theta_Rs': 0.002, 'center_y': 0.5, 'center_x': 0.25, 'Rs': 0.13}
        kwargs_lens_subs = [sub_1, sub_2, sub_3, sub_4, sub_5]
        z_list_subs = [0.6, 0.65, 0.7, 0.7, 0.8]
        lens_model_list_subs = ['NFW'] * 5

        self.kwargs_lens_full_background = kwargs_lens_simple_background + kwargs_lens_subs

        self.lensmodel_fixed_background = LensModel(lens_model_list=lens_model_list_simple_background + lens_model_list_subs,
                          redshift_list=z_list_simple + z_list_subs,z_source=1.5,
                          cosmo=self.cosmo, multi_plane=True)

        self.x_pos_single_background = np.array([0.52879627,-0.51609593,-0.55462914, 0.39140589])
        self.y_pos_single_background = np.array([-0.6484213, 0.54131023, -0.34026707, 0.46996126])

    def test_single_background_nobackground(self):
        lens_model_list_simple = ['SPEP', 'SHEAR']
        z_list_simple = [0.5, 0.5]
        kwargs_lens_simple = [{'theta_E': 0.7, 'center_x': -0.01, 'center_y': 0.001, 'e1': 0.018, 'gamma': 2.,
                               'e2': 0.089}, {'e1': 0.041, 'e2': -0.029}]

        lens_model_list_subs = ['NFW']
        z_list_subs = [0.4]
        sub_1 = {'theta_Rs': 0.01, 'center_y': -0.58, 'center_x': 0.58, 'Rs': 0.08}
        kwargs_lens_subs = [sub_1]

        kwargs_full = kwargs_lens_simple + kwargs_lens_subs
        lens_model_list_full = lens_model_list_simple + lens_model_list_subs
        z_list_full = z_list_simple+ z_list_subs

        srcxtrue, srcytrue = 0.01, -0.04
        xpos = np.array([0.55802196, -0.58764811, -0.52326657, 0.36021833])
        ypos = np.array([-0.60748955, 0.47690715, -0.37781007, 0.47671429])
        xtrue,ytrue = np.array([1056.64474051,-1109.84714442,-988.25453294,680.31748612]), np.array([-1143.88019534,900.69895499,-713.54169304,900.3347147])
        alphaxtrue,alphaytrue = np.array([-0.40925733,0.46601146,0.4168877,-0.25722007]),np.array([0.3916104,-0.4344053,0.21775245,-0.43425814])
        lensmodel = LensModel(lens_model_list=lens_model_list_full, redshift_list=z_list_full,
                              z_source=1.5, cosmo=self.cosmo, multi_plane=True)

        split = MultiPlaneLensing(lensmodel, xpos, ypos, kwargs_full, 1.5, 0.5,
                                  self.cosmo, [0, 1], single_background=True)
        betax, betay = split.ray_shooting_fast(kwargs_full[0:2])

        xout,yout = split._background._fixed_background(xtrue,ytrue,kwargs_full,alphaxtrue,alphaytrue)
        print(xout*split._background._T_z_source**-1,yout*split._background._T_z_source**-1)
        print(betax,betay)

        exit(1)



    def test_single_background(self):

        alpha_x_true = np.array([-0.36907648, 0.41860318, 0.42265953, -0.26021003])
        alpha_y_true = np.array([0.4160718, -0.48592582, 0.17717094, -0.42390834])
        x_in_true = np.array([998.6980648, -974.71188014, -1047.48667913, 739.21910397])
        y_in_true = np.array([-1224.62493444, 1022.33224746, -642.6370298, 887.58077074])
        true_shear_e1, true_shear_e2 = 0.0041, -0.029

        true = MultiPlaneLensing(self.lensmodel_fixed_background, self.x_pos_single_background,
                                  self.y_pos_single_background, self.kwargs_lens_full_background, 1.5, 0.5,
                                  self.cosmo, [0, 1], single_background=False)

        betax_true, betay_true = true.ray_shooting_fast(self.kwargs_lens_full_background[0:2],
                                         thetax=self.x_pos_single_background,thetay=self.y_pos_single_background)

        split = MultiPlaneLensing(self.lensmodel_fixed_background, self.x_pos_single_background,
                                  self.y_pos_single_background, self.kwargs_lens_full_background, 1.5, 0.5,
                                  self.cosmo,[0, 1], single_background=True)

        split._background._alpha_x_approx = alpha_x_true
        split._background._alpha_y_approx = alpha_y_true

        x,y = split._background._fixed_background(x_in_true,y_in_true,self.kwargs_lens_full_background[2:],alpha_x_true,alpha_y_true)

        betax,betay = x*split._T_z_source**-1,y*split._T_z_source**-1

        npt.assert_almost_equal(betax,betax_true)
        npt.assert_almost_equal(betay,betay_true)

    def test_single_background_magfinite(self):

        approx = MultiPlaneLensing(self.lensmodel_fixed_background, self.x_pos_single_background,
                                  self.y_pos_single_background, self.kwargs_lens_full_background, 1.5, 0.5,
                                  self.cosmo, [0, 1], single_background=True)

        mag_point = approx.magnification(self.x_pos_single_background, self.y_pos_single_background,
                                         self.kwargs_lens_full_background)
        print(mag_point)

        extension_approx = LensModelExtensions(approx)

        mag_finite_approx = extension_approx.magnification_finite(self.x_pos_single_background[0:2],
                                                                  self.y_pos_single_background[0:2],
                                                                  self.kwargs_lens_full_background, source_sigma=0.0005,grid_number=200,
                                                                  polar_grid=False,window_size=0.04)
        print(mag_finite_approx)
        #a=input('continue')
        approx._set_background_hessian(self.kwargs_lens_full_background[0:2])

        extension_approx = LensModelExtensions(approx)

        mag_finite_approx = extension_approx.magnification_finite(self.x_pos_single_background[0:2], self.y_pos_single_background[0:2],
                                                           self.kwargs_lens_full_background,source_sigma=0.0005,
                                                                  grid_number=200,polar_grid=False,window_size=0.04)
        print(mag_finite_approx)
        exit(1)

    def test_param_transform(self):

        args = self.optimizer_params._lower_limit

        args_dictionary = self.optimizer_params._params.argstovary_todictionary(args)
        args_array = self.optimizer_params._params._kwargs_to_tovary(args_dictionary)

        npt.assert_allclose(args,args_array)

    def test_penalties(self):

        args = self.optimizer_params._lower_limit

        self.optimizer_params._optimizer._param_penalties(args)

        self.optimizer_params._optimizer._param_penalties(args)

    def test_params(self):

        param_class = self.optimizer_subs._params

        all = self.front_args + self.main_args + self.back_args
        assert param_class.tovary_indicies == [0,1]
        assert param_class.args_tovary == self.kwargs_lens_simple
        assert param_class.args_fixed == self.front_args + self.main_args + self.back_args
        assert param_class.argsfixed_todictionary() == all

    def test_fixed_routines(self):

        sie = FixedPowerLaw_Shear(['SPEMD','SHEAR'],self.kwargs_lens_simple,self.x_pos_simple,self.y_pos_simple)

        assert np.absolute(sie._theta_E_start - 0.7) < 0.2
        for i,group in enumerate(sie.param_names):
            for name in group:
                assert name in self.kwargs_lens_simple[i]

        low,high = sie.get_param_ranges()
        assert len(low) == len(high)

        spep = FixedPowerLaw_Shear(['SPEP', 'SHEAR'],self.kwargs_lens_simple,self.x_pos_simple,self.y_pos_simple)
        assert np.absolute(spep._theta_E_start - 0.7) < 0.2
        for i, group in enumerate(sie.param_names):
            for name in group:
                assert name in self.kwargs_lens_simple[i]

        low, high = spep.get_param_ranges()
        assert len(low) == len(high)

    def test_split_multi_plane_lensmodels(self):

        split = MultiPlaneLensing(self.lens_model_full,self.x_pos_simple,self.y_pos_simple,self.kwargs_lens_full,
                                  1.5,0.5,self.cosmo,[0,1])

        macromodel_lensmodel, macro_args, halos_lensmodel, halos_args,_ = \
            split._split_lensmodel(self.lens_model_full, self.kwargs_lens_full, z_break=0.5, macro_indicies=[0, 1])

        assert macro_args == self.kwargs_lens_simple

        assert halos_args == self.front_args+self.main_args+self.back_args

        fore = split._foreground
        main = split._model_to_vary
        back = split._background

        _ = fore.ray_shooting(split._halo_args, true_foreground=True)

        assert fore._z_to_vary == 0.5
        assert back._z_source == 1.5
        assert main._z_to_vary == 0.5

        output = fore._rays['x'],fore._rays['y'],fore._rays['alphax'],fore._rays['alphay']
        output_true = self.lens_model_front.lens_model.ray_shooting_partial(np.zeros_like(self.x_pos_simple),
                                                                            np.zeros_like(self.y_pos_simple),
                                                                            self.x_pos_simple, self.y_pos_simple, 0, 0.5,
                                                                            self.kwargs_front)

        for (_split,true) in zip(output,output_true):
            npt.assert_almost_equal(_split,true)

    def test_split_multiplane_rayshooting(self):

        model = self.lens_model_full

        kwargs = self.kwargs_lens_full

        xpos,ypos = self.x_pos_simple, self.y_pos_simple

        split = MultiPlaneLensing(model, xpos, ypos, kwargs,
                                  1.5, 0.5, self.cosmo, [0, 1])

        betax_true, betay_true = model.ray_shooting(xpos, ypos, kwargs)

        args = self.kwargs_lens_full[0:2]
        betax, betay = split.ray_shooting_fast(macromodel_args=args,thetax=xpos,thetay=ypos,force_compute=True)
        betax_fast, betay_fast = split.ray_shooting_fast(split._macro_args)

        betax_func, betay_func = split.ray_shooting(0, 0.5, self.kwargs_lens_full)
        betax_true_2, betay_true_2 = self.lens_model_full.ray_shooting(0, 0.5, self.kwargs_lens_full)

        npt.assert_almost_equal(betax_func,betax_true_2)
        npt.assert_almost_equal(betay_func,betay_true_2)

        npt.assert_almost_equal(betax, betax_fast)
        npt.assert_almost_equal(betax_true, betax)
        npt.assert_almost_equal(betay_true, betay)
        npt.assert_almost_equal(betax_true, betax_fast)
        npt.assert_almost_equal(betay_true, betay_fast)

    def test_split_multiplane_hessian(self):

        split = MultiPlaneLensing(self.lens_model_full, self.x_pos_simple, self.y_pos_simple, self.kwargs_lens_full,
                                  1.5, 0.5, self.cosmo, [0, 1])

        _  = split.ray_shooting_fast(self.kwargs_lens_full[0:2])

        f1,f2,f3,f4 = split.hessian_fast(split._macro_args)
        t1,t2,t3,t4 = self.lens_model_full.hessian(self.x_pos_simple,self.y_pos_simple,self.kwargs_lens_full)

        npt.assert_almost_equal(f1,t1)
        npt.assert_almost_equal(f2,t2)
        npt.assert_almost_equal(f3, t3)
        npt.assert_almost_equal(f4, t4)

        f1, f2, f3, f4 = split.hessian(0.5,0.5,self.kwargs_lens_full)
        t1, t2, t3, t4 = self.lens_model_full.hessian(0.5,0.5, self.kwargs_lens_full)

        npt.assert_almost_equal(f1, t1)
        npt.assert_almost_equal(f2, t2)
        npt.assert_almost_equal(f3, t3)
        npt.assert_almost_equal(f4, t4)

    def test_split_multi_plane_magnification(self):

        split = MultiPlaneLensing(self.lens_model_full, self.x_pos_simple, self.y_pos_simple, self.kwargs_lens_full,
                                  1.5, 0.5, self.cosmo, [0, 1])

        _ = split.ray_shooting_fast(self.kwargs_lens_full[0:2])

        magnification_true = np.absolute(self.lens_model_full.magnification(self.x_pos_simple,
                                                                            self.y_pos_simple,self.kwargs_lens_full))
        magnification_split = split.magnification_fast(self.kwargs_lens_simple)

        npt.assert_almost_equal(magnification_true*max(magnification_true)**-1,
                                magnification_split*max(magnification_split)**-1,2)

        mag_true = self.lens_model_full.magnification(np.array([0,0.2]),np.array([0.4,0.6]),self.kwargs_lens_full)
        mag_true_split = split.magnification(np.array([0,0.2]),np.array([0.4,0.6]),self.kwargs_lens_full)

        npt.assert_almost_equal(mag_true,mag_true_split)

    def test_multi_plane_simple(self):

        kwargs_lens, source, [x_image, y_image], _ = self.optimizer_simple.optimize(n_particles=10, n_iterations=10, restart=2)
        _ = self.optimizer_simple.lensModel.magnification(x_image, y_image, kwargs_lens)

        self.optimizer_simple._tol_src_penalty = 1e-30

        kwargs_lens, source, [x_image, y_image], _ = self.optimizer_simple.optimize(n_particles=10, n_iterations=10,
                                                                                 restart=2)
        _ = self.optimizer_simple.lensModel.magnification(x_image, y_image, kwargs_lens)

    def test_multi_plane_reoptimize(self, tol=0.004):


        lens_model_list_reoptimize = self.lens_model_full.lens_model_list
        redshift_list_reoptimize = self.lens_model_full.redshift_list

        for val in [True,False]:
            reoptimizer = Optimizer(self.x_pos_simple, self.y_pos_simple,
                                    magnification_target=self.magnification_simple,
                                    redshift_list=redshift_list_reoptimize,
                                    lens_model_list=lens_model_list_reoptimize, kwargs_lens=self.kwargs_lens_full, multiplane=True,
                                    verbose=True, z_source=1.5, z_main=0.5, astropy_instance=self.cosmo,
                                    optimizer_routine='fixed_powerlaw_shear', re_optimize=True, particle_swarm=val)

            kwargs_lens, source, [x_image, y_image], _ = reoptimizer.optimize(n_particles=20, n_iterations=10, restart=2)
            _ = reoptimizer.lensModel.magnification(x_image, y_image, kwargs_lens)

    def test_multi_plane_subs(self,tol=0.004):

        kwargs_lens, source, [x_image,y_image] = self.optimizer_subs.optimize(n_particles=20, n_iterations=10, restart=2)
        # this should just finish with no errors raised

t = TestMultiPlaneOptimizer()
t.setup()
t.test_single_background_nobackground()


if __name__ == '__main__':
    pytest.main()
