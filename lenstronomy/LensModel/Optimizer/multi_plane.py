from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.Optimizer.split_multiplane import SplitMultiplane as split_multi
import numpy as np


class MultiPlaneOptimizer(object):

    def __init__(self, lensmodel_full, all_args, x_pos, y_pos, tol_source, Params, magnification_target,
                 tol_mag, centroid_0, tol_centroid, z_main, z_src, astropy_instance, interpolated,return_mode = 'PSO',
                 mag_penalty=False,return_array = False, verbose=False,
                  pso_convergence_mean=None,pso_compute_magnification=None):

        self.Params = Params
        self.lensModel = lensmodel_full
        self.all_lensmodel_args = all_args
        self.solver = LensEquationSolver(self.lensModel)

        self.tol_source = tol_source

        self.magnification_target = magnification_target
        self.tol_mag = tol_mag

        self._compute_mags_flag = mag_penalty

        self._return_array = return_array

        self._pso_convergence_mean = pso_convergence_mean
        self._pso_compute_magnification = pso_compute_magnification

        self.centroid_0 = centroid_0
        self.tol_centroid = tol_centroid

        self.verbose = verbose

        self._x_pos,self._y_pos = np.array(x_pos),np.array(y_pos)
        self.mag_penalty,self.src_penalty,self.parameters = [],[], []

        self.multiplane_optimizer = split_multi(x_pos,y_pos,lensmodel_full,all_args,interpolated=interpolated,
                                                    z_source=z_src,z_macro=z_main,astropy_instance=astropy_instance,verbose=verbose,
                                                    macro_indicies=self.Params.tovary_indicies)

        self._return_mode = return_mode

        self.reset()

    def reset(self):

        self.mag_penalty, self.src_penalty, self.centroid_penalty, self.parameters = [], [], [], []
        self._converged = False
        self._counter = 1
        self._compute_mags = self._compute_mags_flag
        self.is_converged = False

    def get_best(self):

        total = np.array(self.src_penalty) + np.array(self.mag_penalty) + np.array(self.centroid_penalty)

        return total[np.argmin(total)]

    def _init_particles(self,n_particles,n_iterations):

        self._n_total_iter = n_iterations*n_particles
        self._n_particles = n_particles
        self._mag_penalty_switch = 1

    def _get_images(self,kwargs_varied):

        srcx, srcy = self.multiplane_optimizer.ray_shooting_fast(kwargs_varied)

        args = kwargs_varied + self.Params.argsfixed_todictionary()

        source_x, source_y = np.mean(srcx), np.mean(srcy)
        x_image, y_image = self.solver.image_position_from_source(source_x, source_y, args, precision_limit=10**-10)

        return x_image, y_image, source_x, source_y

    def _source_position_penalty(self, lens_args_tovary):

        betax,betay = self.multiplane_optimizer.ray_shooting_fast(lens_args_tovary)

        dx = ((betax[0] - betax[1]) ** 2 + (betax[0] - betax[2]) ** 2 + (betax[0] - betax[3]) ** 2 + (
                betax[1] - betax[2]) ** 2 +
              (betax[1] - betax[3]) ** 2 + (betax[2] - betax[3]) ** 2)
        dy = ((betay[0] - betay[1]) ** 2 + (betay[0] - betay[2]) ** 2 + (betay[0] - betay[3]) ** 2 + (
                betay[1] - betay[2]) ** 2 +
              (betay[1] - betay[3]) ** 2 + (betay[2] - betay[3]) ** 2)

        if self._return_array:
            return 0.5 * np.array([dx,dy]) * self.tol_source ** -2
        else:
            return 0.5 * (dx + dy) * self.tol_source ** -2

    def _magnification_penalty(self, lens_args,magnification_target, tol):

        magnifications = self.multiplane_optimizer.magnification_fast(lens_args)

        magnifications *= max(magnifications) ** -1

        dM = []

        for i, target in enumerate(magnification_target):
            mag_tol = tol * target
            dM.append((magnifications[i] - target) * mag_tol ** -1)

        dM = np.array(dM)

        if self._return_array:
            return 0.5*dM**2
        else:
            return 0.5 * np.sum(dM ** 2)

    def _centroid_penalty(self, values_dic, tol_centroid):

        d_centroid = ((values_dic[0]['center_x'] - self.centroid_0[0]) * tol_centroid ** -1) ** 2 + \
                     ((values_dic[0]['center_y'] - self.centroid_0[1]) * tol_centroid ** -1) ** 2

        return 0.5 * d_centroid

    def _log(self,src_penalty,mag_penalty,centroid_penalty):

        if mag_penalty is None:
            mag_penalty = np.inf
        if src_penalty is None:
            src_penalty = np.inf
        if centroid_penalty is None:
            centroid_penalty = np.inf

        self.src_penalty.append(np.sum(src_penalty))
        self.mag_penalty.append(np.sum(mag_penalty))
        self.centroid_penalty.append(np.sum(centroid_penalty))
        self.parameters.append(self.lens_args_latest)

    def _compute_mags_criterion(self):

        if self._compute_mags:
            return True

        if self._counter > self._n_particles and np.mean(self.src_penalty[-self._n_particles:]) < \
                self._pso_compute_magnification:
            return True
        else:
            return False

    def _test_convergence(self):

        if self._counter <= self._n_particles:
            self.is_converged = False
            return

        if np.mean(self.src_penalty[-self._n_particles:]) < self._pso_convergence_mean:
            self.is_converged = True
        else:
            self.is_converged = False

    def __call__(self, lens_values_tovary, src_penalty=None,mag_penalty=None,centroid_penalty=None):

        self._counter += 1

        params_fixed = self.Params.argsfixed_todictionary()
        lens_args_tovary = self.Params.argstovary_todictionary(lens_values_tovary)

        if self.tol_source is not None:

            src_penalty = self._source_position_penalty(lens_args_tovary)

            self._compute_mags = self._compute_mags_criterion()

        if self._compute_mags and self.tol_mag is not None:
            mag_penalty = self._magnification_penalty(lens_args_tovary,self.magnification_target,self.tol_mag)

        if self.tol_centroid is not None:
            centroid_penalty = self._centroid_penalty(lens_args_tovary,self.tol_centroid)

        if self._return_array:
            penalty = src_penalty

            if self._compute_mags and self.tol_mag is not None:
                penalty = np.append(penalty,mag_penalty)

            if self.tol_centroid is not None:
                penalty = np.append(penalty,centroid_penalty)

        else:
            _penalty = [src_penalty,mag_penalty,centroid_penalty]

            penalty = 0
            for pen in _penalty:
                if pen is not None:
                    penalty += pen

        if self._counter % 500 == 0 and self.verbose:

            print('source penalty: ', src_penalty)
            if self.mag_penalty is not None:
                print('mag penalty: ', mag_penalty)

        self.lens_args_latest = lens_args_tovary + params_fixed

        self._log(src_penalty,mag_penalty,centroid_penalty)

        self._test_convergence()

        if self._return_mode == 'PSO':
            return -1 * penalty, None
        else:
            return np.array(penalty)
