from lenstronomy.LensModel.Optimizer.optimizer import Optimizer, MultiPlaneLensing
import numpy.testing as npt
from astropy.cosmology import FlatLambdaCDM
import pytest
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Optimizer.fixed_routines import *

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
        #self.kwargs_lens_simple = [
        #    {'theta_E': 0., 'center_x': 0.0, 'center_y': 0, 'e1': 0.0185665252864011, 'gamma': 2.,
        #     'e2': 0.08890716633399057}, {'e1': 0, 'e2': 0}]

        front_halos = ['NFW', 'SIS', 'NFW', 'NFW', 'NFW', 'NFW']
        front_redshifts = [0.4, 0.4, 0.4, 0.44, 0.44, 0.44]
        self.front_args = [{'alpha_Rs': 0.001, 'center_y': 0.2, 'center_x': 1.2, 'Rs': 0.13},
                           {'theta_E': 0.4, 'center_y': -0.2, 'center_x': 0.7},
                           {'alpha_Rs': 0.004, 'center_y': 0.12, 'center_x': -1.2, 'Rs': 0.13},
                           {'alpha_Rs': 0.0001, 'center_y': 0.32, 'center_x': -0.2, 'Rs': 0.13},
                           {'alpha_Rs': 0.003, 'center_y': 0.82, 'center_x': 0.78, 'Rs': 0.13},
                           {'alpha_Rs': 0.008, 'center_y': 1, 'center_x': 0.75, 'Rs': 0.16}]

        main_halos = ['NFW', 'NFW', 'NFW', 'SIS', 'NFW', 'NFW']
        main_redshifts = [0.5] * 6
        self.main_args = [{'alpha_Rs': 0.001, 'center_y': 1.2, 'center_x': 0.2, 'Rs': 0.13},
                          {'alpha_Rs': 0.002, 'center_y': -0.1, 'center_x': 0.91, 'Rs': 0.11},
                          {'alpha_Rs': 0.009, 'center_y': 0.18, 'center_x': -0.42, 'Rs': 0.13},
                          {'theta_E': 0.3, 'center_y': 0.42, 'center_x': -0.92},
                          {'alpha_Rs': 0.005, 'center_y': 0.9, 'center_x': 0.48, 'Rs': 0.13},
                          {'alpha_Rs': 0.008, 'center_y': -1, 'center_x': 0.95, 'Rs': 0.16}]


        back_halos = ['NFW', 'NFW', 'NFW', 'NFW', 'SIS', 'NFW']
        back_redshifts = [0.55, 0.6, 0.6, 0.74, 0.74, 0.8]
        self.back_args = [{'alpha_Rs': 0.004, 'center_y': 0.1, 'center_x': 1, 'Rs': 0.13},
                          {'alpha_Rs': 0.001, 'center_y': 0.2, 'center_x': 0.7, 'Rs': 0.11},
                          {'alpha_Rs': 0.003, 'center_y': -0.1, 'center_x': -1, 'Rs': 0.13},
                          {'alpha_Rs': 0.0008, 'center_y': 0.42, 'center_x': 0.1, 'Rs': 0.13},
                          {'theta_E': 0.3, 'center_y': 0.7, 'center_x': 1.08},
                          {'alpha_Rs': 0.006, 'center_y': 0.5, 'center_x': 0.75, 'Rs': 0.16}]

        self.convention_idx = [14,18]
        #self.convention_idx = False

        lens_model_list_full = lens_model_list_simple + front_halos + main_halos + back_halos
        redshift_list_full = redshift_list_simple + front_redshifts + main_redshifts + back_redshifts
        self.kwargs_lens_full = self.kwargs_lens_simple + self.front_args + self.main_args + self.back_args

        self.lens_model_full = LensModel(lens_model_list_full, z_source=1.5, lens_redshift_list=redshift_list_full,
                                         cosmo=self.cosmo,
                                         multi_plane=True)
        self.lens_model_full_convention = LensModel(lens_model_list_full, z_source=1.5, lens_redshift_list=redshift_list_full,
                                         cosmo=self.cosmo,
                                         multi_plane=True, observed_convention_index=self.convention_idx)

        self.lens_model_front = LensModel(front_halos + main_halos, lens_redshift_list=front_redshifts + main_redshifts,
                                          z_source=1.5,
                                          cosmo=self.cosmo, multi_plane=True)
        self.kwargs_front = self.front_args + self.main_args

        self.lens_model_simple = LensModel(lens_model_list_simple, z_source=1.5, lens_redshift_list=redshift_list_simple,
                                           cosmo=self.cosmo,
                                           multi_plane=True)

        self.optimizer_simple = Optimizer(self.x_pos_simple, self.y_pos_simple,
                                          magnification_target=self.magnification_simple,
                                          redshift_list=redshift_list_simple, simplex_n_iterations=2,
                                          lens_model_list=lens_model_list_simple, kwargs_lens=self.kwargs_lens_simple,
                                          multiplane=True, verbose=True, z_source=1.5, z_main=0.5,
                                          astropy_instance=self.cosmo, optimizer_routine='fixed_powerlaw_shear',
                                          tol_simplex_func=1e+9, pso_convergence_mean=1e+9,)

        self.optimizer_subs = Optimizer(self.x_pos_simple, self.y_pos_simple,
                                        magnification_target=self.magnification_simple, pso_convergence_mean=1e+9,
                                        redshift_list=redshift_list_full, simplex_n_iterations=2,tol_simplex_func=1e+9,
                                        lens_model_list=lens_model_list_full, kwargs_lens=self.kwargs_lens_full,
                                        multiplane=True, verbose=True, z_source=1.5, z_main=0.5,
                                        astropy_instance=self.cosmo,optimizer_routine='fixed_powerlaw_shear')

        self.optimizer_subs_convention = Optimizer(self.x_pos_simple, self.y_pos_simple,
                                        magnification_target=self.magnification_simple, pso_convergence_mean=1e+9,
                                        redshift_list=redshift_list_full, simplex_n_iterations=2, tol_simplex_func=1e+9,
                                        lens_model_list=lens_model_list_full, kwargs_lens=self.kwargs_lens_full,
                                        multiplane=True, verbose=True, z_source=1.5, z_main=0.5,
                                        astropy_instance=self.cosmo, optimizer_routine='fixed_powerlaw_shear', observed_convention_index=self.convention_idx)

        self.optimizer_params = Optimizer(self.x_pos_simple, self.y_pos_simple,
                                        magnification_target=self.magnification_simple, tol_simplex_func=1e+9, pso_convergence_mean=1e+9,
                                        redshift_list=redshift_list_full, simplex_n_iterations=2,
                                        lens_model_list=lens_model_list_full, kwargs_lens=self.kwargs_lens_full,
                                        multiplane=True, verbose=True, z_source=1.5, z_main=0.5,
                                        astropy_instance=self.cosmo,optimizer_routine='fixed_powerlaw_shear',
                                        constrain_params={'shear':[0.06,0.01],'shear_pa':[-30,10],'theta_E':[1,0.1]})

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

        sie = FixedPowerLaw_Shear(['SPEMD','SHEAR'],self.kwargs_lens_simple,self.x_pos_simple,self.y_pos_simple,
                                  constrain_params={'shear':[0.06, 0.01]})

        powerlaw = VariablePowerLaw_Shear(['SPEMD','SHEAR'],self.kwargs_lens_simple,self.x_pos_simple,self.y_pos_simple,
                                          constrain_params={'shear': [0.06, 0.01]})

        models = [sie, powerlaw]

        for model in models:

            assert np.absolute(model._theta_E_start - 0.7) < 0.2

            for i,group in enumerate(model.param_names):
                for name in group:
                    assert name in self.kwargs_lens_simple[i]

            low,high = model.get_param_ranges()
            assert len(low) == len(high)

        spep = FixedPowerLaw_Shear(['SPEP', 'SHEAR'],self.kwargs_lens_simple,self.x_pos_simple,self.y_pos_simple,
                                   constrain_params={'shear': [0.06, 0.01]})
        assert np.absolute(spep._theta_E_start - 0.7) < 0.2
        for i, group in enumerate(sie.param_names):
            for name in group:
                assert name in self.kwargs_lens_simple[i]

        low, high = spep.get_param_ranges()
        assert len(low) == len(high)

    def test_split_multi_plane_lensmodels(self):

        split = MultiPlaneLensing(self.lens_model_full,self.x_pos_simple,self.y_pos_simple,self.kwargs_lens_full,
                                  1.5,0.5,self.cosmo,[0,1], {}, None)

        macromodel_lensmodel, macro_args, halos_lensmodel, halos_args = \
            split._split_lensmodel(self.lens_model_full, self.kwargs_lens_full, [0, 1], None, self.convention_idx)

        assert macro_args == self.kwargs_lens_simple

        assert halos_args == self.front_args+self.main_args+self.back_args

        fore = split._foreground

        _ = fore.ray_shooting(split._halo_args, macro_args, offset_index=0, force_compute=False)

        assert fore._z_to_vary == 0.5

        output = fore._rays[0]['x'],fore._rays[0]['y'],fore._rays[0]['alphax'],fore._rays[0]['alphay']
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
                                  1.5, 0.5, self.cosmo, [0, 1], {}, None)

        betax_true, betay_true = model.ray_shooting(xpos, ypos, kwargs)

        args = self.kwargs_lens_full[0:2]

        betax, betay = split._ray_shooting_fast(macromodel_args=args,
                                                thetax=xpos, thetay=ypos, force_compute=True)

        betax_func, betay_func = split.ray_shooting(0, 0.5, self.kwargs_lens_full)
        betax_true_2, betay_true_2 = self.lens_model_full.ray_shooting(0, 0.5, self.kwargs_lens_full)

        npt.assert_almost_equal(betax_func,betax_true_2)
        npt.assert_almost_equal(betay_func,betay_true_2)

        npt.assert_almost_equal(betax_true, betax)
        npt.assert_almost_equal(betay_true, betay)
        npt.assert_almost_equal(betax_true, betax)
        npt.assert_almost_equal(betay_true, betay)

    def test_split_multiplane_hessian(self):

        split = MultiPlaneLensing(self.lens_model_full, self.x_pos_simple, self.y_pos_simple, self.kwargs_lens_full,
                                  1.5, 0.5, self.cosmo, [0, 1], {}, None)

        _ = split._ray_shooting_fast(self.kwargs_lens_full[0:2])

        f1,f2,f3,f4 = split._hessian_fast(self.kwargs_lens_simple)
        t1,t2,t3,t4 = self.lens_model_full.hessian(self.x_pos_simple,self.y_pos_simple,self.kwargs_lens_full)

        npt.assert_almost_equal(f1,t1, decimal=6)
        npt.assert_almost_equal(f2,t2, decimal=6)
        npt.assert_almost_equal(f3, t3, decimal=6)
        npt.assert_almost_equal(f4, t4, decimal=6)

        f1, f2, f3, f4 = split.hessian(0.5,0.5,self.kwargs_lens_full)
        t1, t2, t3, t4 = self.lens_model_full.hessian(0.5,0.5, self.kwargs_lens_full)

        npt.assert_almost_equal(f1, t1)
        npt.assert_almost_equal(f2, t2)
        npt.assert_almost_equal(f3, t3)
        npt.assert_almost_equal(f4, t4)

    def test_split_multi_plane_magnification(self):

        split = MultiPlaneLensing(self.lens_model_full, self.x_pos_simple, self.y_pos_simple, self.kwargs_lens_full,
                                  1.5, 0.5, self.cosmo, [0, 1], {}, None)

        _ = split._ray_shooting_fast(self.kwargs_lens_full[0:2])

        magnification_true = np.absolute(self.lens_model_full.magnification(self.x_pos_simple,
                                                                            self.y_pos_simple,self.kwargs_lens_full))
        magnification_split = split._magnification_fast(self.kwargs_lens_simple)

        npt.assert_almost_equal(magnification_true*max(magnification_true)**-1,
                                magnification_split*max(magnification_split)**-1,2)

        mag_true = self.lens_model_full.magnification(np.array([0,0.2]),np.array([0.4,0.6]),self.kwargs_lens_full)
        mag_true_split = split.magnification(np.array([0,0.2]),np.array([0.4,0.6]),self.kwargs_lens_full)

        npt.assert_almost_equal(mag_true, mag_true_split, decimal=4)

    def test_multi_plane_simple(self):

        kwargs_lens, source, [x_image,y_image] = self.optimizer_simple.optimize(n_particles=2, n_iterations=2, restart=2)
        _ = self.optimizer_simple._lensModel.magnification(x_image, y_image, kwargs_lens)

        self.optimizer_simple._tol_src_penalty = 1e-30

        kwargs_lens, source, [x_image, y_image] = self.optimizer_simple.optimize(n_particles=2, n_iterations=2,
                                                                                 restart=2)
        _ = self.optimizer_simple._lensModel.magnification(x_image, y_image, kwargs_lens)

    def test_multi_plane_reoptimize(self, tol=0.004):

        lens_model_list_reoptimize = self.lens_model_full.lens_model_list
        redshift_list_reoptimize = self.lens_model_full.redshift_list
        scale = [0.5, 1]

        routine = ['fixed_powerlaw_shear','variable_powerlaw_shear', 'fixed_powerlaw_shear', 'variable_powerlaw_shear',
                   'fixedshearpowerlaw']
        constrainparams = [None, None, {'shear': [0.06, 1]}, {'shear': [0.06, 1]}, {'shear': 0.06}]

        for ri, rout in enumerate(routine):
            for i, val in enumerate([False,True]):
                reoptimizer = Optimizer(self.x_pos_simple, self.y_pos_simple,
                                        magnification_target=self.magnification_simple,
                                        redshift_list=redshift_list_reoptimize,
                                        lens_model_list=lens_model_list_reoptimize, kwargs_lens=self.kwargs_lens_full, multiplane=True,
                                        verbose=True, z_source=1.5, z_main=0.5, astropy_instance=self.cosmo, tol_simplex_func=1e+9,
                                        pso_convergence_mean=1e+9, optimizer_routine=rout, re_optimize=True, particle_swarm=val, simplex_n_iterations=2,
                                        optimizer_kwargs={'re_optimize_scale': scale[i], 'save_background_path': val}, constrain_params=constrainparams[ri])

                kwargs_lens, source, [x_image, y_image] = reoptimizer.optimize(n_particles=2, n_iterations=2, restart=2)

                _ = reoptimizer._lensModel.magnification(x_image, y_image, kwargs_lens)

        optimized_kwargs = {'precomputed_rays': reoptimizer.lensModel._foreground._rays, 'optimization_algorithm': 'powell'}
        reoptimizer = Optimizer(self.x_pos_simple, self.y_pos_simple,
                                magnification_target=self.magnification_simple,
                                redshift_list=redshift_list_reoptimize,
                                lens_model_list=lens_model_list_reoptimize, kwargs_lens=self.kwargs_lens_full,
                                multiplane=True, simplex_n_iterations=2, tol_simplex_func=1e+9,
                                verbose=True, z_source=1.5, z_main=0.5, astropy_instance=self.cosmo,
                                optimizer_routine='fixed_powerlaw_shear', re_optimize=True, particle_swarm=False,
                                optimizer_kwargs=optimized_kwargs, compute_mags_postpso=True, tol_mag=None)

        kwargs_lens, source, [x_image, y_image] = reoptimizer.optimize(n_particles=2,n_iterations=2, restart=2)

    def test_multi_plane_subs(self,tol=0.004):

        kwargs_lens, source, [x_image,y_image] = self.optimizer_subs.optimize(n_particles=2, n_iterations=2, restart=2)
        # this should just finish with no errors raised

    def test_convention(self):

        lensModel_opt = self.optimizer_subs_convention._optimizer.lensing

        lensModel_observed = self.lens_model_full_convention

        betaxtrue, betaytrue = lensModel_observed.ray_shooting(self.x_pos_simple[0], self.y_pos_simple[0], self.kwargs_lens_full)

        betax, betay = lensModel_opt._ray_shooting_fast(self.kwargs_lens_full[0:2], offset_index=0,
                                                        thetax=self.x_pos_simple[0], thetay=self.y_pos_simple[0], force_compute=True)

        npt.assert_almost_equal(betax, betaxtrue, 5)
        npt.assert_almost_equal(betay, betaytrue, 5)
if __name__ == '__main__':
    pytest.main()
