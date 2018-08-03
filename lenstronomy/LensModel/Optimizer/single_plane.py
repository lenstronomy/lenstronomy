import numpy as np
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

class SinglePlaneOptimizer(object):

    def __init__(self, lensmodel, x_pos, y_pos, tol_source, params, \
                 magnification_target, tol_mag, centroid_0, tol_centroid, k_start=0, arg_list=[],
                 return_mode='PSO',verbose=False,mag_penalty=False):

        self.Params = params
        self.lensModel = lensmodel
        self.solver = LensEquationSolver(self.lensModel)

        self.tol_source = tol_source

        self.magnification_target = magnification_target
        self.tol_mag = tol_mag
        self._compute_mags_flag = mag_penalty

        self.centroid_0 = centroid_0
        self.tol_centroid = tol_centroid

        self._x_pos, self._y_pos = np.array(x_pos), np.array(y_pos)

        self.verbose=verbose

        self.all_lensmodel_args = arg_list

        self._return_mode = return_mode

        # compute the foreground deflections and second derivatives from subhalos
        if k_start > 0 and len(arg_list)>k_start:

            self._k_sub = np.arange(k_start, len(arg_list))
            self._k_macro = np.arange(0, k_start)

            # subhalo deflections
            self.alpha_x_sub, self.alpha_y_sub = self.lensModel.alpha(x_pos,y_pos,arg_list,self._k_sub)

            # subhalo hessian components
            if tol_mag is not None:
                self.sub_fxx, self.sub_fxy, _, self.sub_fyy = self.lensModel.hessian(x_pos,y_pos,arg_list,self._k_sub)

        else:

            self._k_macro, self._k_sub = None, None
            self.alpha_x_sub, self.alpha_y_sub = 0, 0
            self.sub_fxx, self.sub_fyy, self.sub_fxy = 0,0,0

        self.reset()

    def reset(self):

        self.mag_penalty, self.src_penalty, self.parameters = [], [], []
        self._counter = 1
        self._compute_mags = self._compute_mags_flag
        self.is_converged = False

    def get_best(self):

        total = np.array(self.src_penalty) + np.array(self.mag_penalty)

        return total[np.argmin(total)]

    def _init_particles(self,n_particles,n_iterations):

        self._n_total_iter = n_iterations*n_particles
        self._n_particles = n_particles
        self._mag_penalty_switch = 1

    def _get_images(self,kwargs_varied):

        args = kwargs_varied + self.Params.argsfixed_todictionary()

        srcx, srcy = self.lensModel.ray_shooting(self._x_pos, self._y_pos, args, None)

        source_x, source_y = np.mean(srcx), np.mean(srcy)

        x_image, y_image = self.solver.findBrightImage(source_x, source_y,args,precision_limit=10**-10)

        return x_image, y_image, source_x, source_y

    def _source_position_penalty(self, lens_args):

        # compute the macromodel deflection
        alphax_macro,alphay_macro = self.lensModel.alpha(self._x_pos,self._y_pos,lens_args,k=self._k_macro)

        # compute the source position
        betax = self._x_pos - alphax_macro - self.alpha_x_sub
        betay = self._y_pos - alphay_macro - self.alpha_y_sub

        dx = ((betax[0] - betax[1]) ** 2 + (betax[0] - betax[2]) ** 2 + (betax[0] - betax[3]) ** 2 + (
                    betax[1] - betax[2]) ** 2 +
              (betax[1] - betax[3]) ** 2 + (betax[2] - betax[3]) ** 2)
        dy = ((betay[0] - betay[1]) ** 2 + (betay[0] - betay[2]) ** 2 + (betay[0] - betay[3]) ** 2 + (
                    betay[1] - betay[2]) ** 2 +
              (betay[1] - betay[3]) ** 2 + (betay[2] - betay[3]) ** 2)

        return 0.5*(dx+dy)*self.tol_source**-2

    def _magnification_penalty(self,  args, magnification_target, tol=0.1):

        fxx_macro,fxy_macro,_,fyy_macro = self.lensModel.hessian(self._x_pos, self._y_pos, args, k=self._k_macro)

        fxx = fxx_macro + self.sub_fxx
        fyy = fyy_macro + self.sub_fyy
        fxy = fxy_macro + self.sub_fxy

        det_J = (1-fxx)*(1-fyy) - fxy**2

        magnifications = np.absolute(det_J**-1)

        magnifications *= max(magnifications)**-1

        dM = []

        for i, target in enumerate(magnification_target):
            mag_tol = tol * target
            dM.append((magnifications[i] - target) * mag_tol ** -1)

        dM = np.array(dM)

        return 0.5*np.sum(dM ** 2)

    def _centroid_penalty(self, values_dic, tol_centroid):

        dx = (values_dic[0]['center_x'] - self.centroid_0[0])*self.tol_centroid**-1
        dy = (values_dic[0]['center_y'] - self.centroid_0[1])*self.tol_centroid**-1

        return 0.5*(dx**2+dy**2)

    def _log(self,src_penalty,mag_penalty):

        if mag_penalty is None:
            mag_penalty = np.inf
        if src_penalty is None:
            src_penalty = np.inf

        self.src_penalty.append(src_penalty)
        self.mag_penalty.append(mag_penalty)
        self.parameters.append(self.lens_args_latest)

    def _compute_mags_criterion(self):

        if self._compute_mags:
            return True

        if self._counter > self._n_particles and np.mean(self.src_penalty[-self._n_particles:]) < 5:
            return True
        else:
            return False

    def _test_convergence(self):

        if self._counter > self._n_particles and np.mean(self.src_penalty[-self._n_particles:]) < 1:
            self.is_converged = True
        else:
            self.is_converged = False

    def __call__(self, lens_values_tovary,src_penalty=None,mag_penalty=None,centroid_penalty=None):

        self._counter += 1

        params_fixed = self.Params.argsfixed_todictionary()
        lens_args_tovary = self.Params.argstovary_todictionary(lens_values_tovary)

        if self.tol_source is not None:
            src_penalty = self._source_position_penalty(lens_args_tovary+params_fixed)

            self._compute_mags = self._compute_mags_criterion()

        if self._compute_mags and self.tol_mag is not None:
            mag_penalty = self._magnification_penalty(lens_args_tovary + params_fixed, self.magnification_target,
                                                      self.tol_mag)

        if self.tol_centroid is not None:
            centroid_penalty = self._centroid_penalty(lens_args_tovary, self.tol_centroid)

        _penalty = [src_penalty, mag_penalty, centroid_penalty]

        penalty = 0
        for pen in _penalty:
            if pen is not None:
                penalty += pen

        if self._counter % 500 == 0 and self.verbose:

            print('source penalty: '), src_penalty
            if self.mag_penalty is not None:
                print('mag penalty: '), mag_penalty

        self.lens_args_latest = lens_args_tovary + params_fixed

        self._log(src_penalty, mag_penalty)

        self._test_convergence()

        if self._return_mode == 'PSO':
            return -1 * penalty, None
        else:
            return penalty