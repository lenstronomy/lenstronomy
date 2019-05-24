import numpy as np
from lenstronomy.Util.param_util import cart2polar,polar2cart
from lenstronomy.Util.util import sort_image_index

class Penalties(object):

    def __init__(self, tol_source, tol_mag, tol_centroid, lensing, centroid_0, magnification_target=None,
                 params_to_constrain=None, param_class=None, pso_convergence_mean = None, pso_compute_magnification=None,
                 compute_mags=False, verbose=False, chi2_mode = 'source', tol_image=None, solver = None):
        """
        This class calls the mutli/single plane lensing classes to do all the high level lensing computations in the
        optimization. It also logs things like the source position penalties, magnifiction penalities, centroid penalities,
        and any additional parameter penalties specific by the user.
        :param tol_image: tolerance on recovered source position, or the image positions if chi2_mode = 'image'
        :param tol_mag: flux uncertainty
        :param tol_centroid: mass centroid uncertainty
        :param lensing: the instance of multi/single plane lensing used to do the ray shooting
        :param centroid_0: best guess mass centroid
        :param magnification_target: the magnifications to optimize for
        :param params_to_constrain: additional parameters to constrain (see documentation in 'Optimizer' class)
        :param param_class: isntance of Params class
        :param pso_convergence_mean: convergence criterion
        :param pso_compute_magnification: when to start computing magnifications
        :param compute_mags: user shouldn't touch this
        :param verbose: print things
        :param chi2_mode: flag to use the image/source plane chi^2 in the optimizer. Image plane can be slow
        'image' -> image plane chi^2, 'source' -> source plane chi^2
        :param solver: only used if chi2_mode == 'image', solves the lens equation to determine image positions
        """

        self.tol_source = tol_source
        if chi2_mode == 'source':

            assert isinstance(self.tol_source, float) or isinstance(self.tol_source, int)
            self._chi_mode = 0

        elif chi2_mode == 'image':
            self.tol_image = tol_image
            self._chi_mode = 1
            self._solver = solver
            self._ximage = lensing._x_pos
            self._yimage = lensing._y_pos
            if isinstance(self.tol_image, list) or isinstance(self.tol_image, np.ndarray):
                assert len(self.tol_image == len(self._ximage))

        else:
            raise Exception("chi2_mode must be either 'source' or 'image'")

        if tol_mag is None:
            self.tol_mag = None
        else:
            if not np.logical_or(isinstance(tol_mag,list),isinstance(tol_mag,np.ndarray)):
                tol_mag = [tol_mag]*4
            self.tol_mag = tol_mag

        self.magnification_target = magnification_target
        self.tol_centroid = tol_centroid
        self.lensing = lensing
        self.centroid_0 = centroid_0

        self._pso_convergence_mean = pso_convergence_mean
        self._pso_compute_magnification = pso_compute_magnification

        self._counter = 0
        self.verbose = verbose

        self.param_class = param_class

        self.params_to_constrain = params_to_constrain

        self._reset(compute_mags)

    def __call__(self,lens_args_to_vary_array):

        mag_penalty, centroid_penalty, param_penalties = None, None, None

        params_fixed = self.param_class.argsfixed_todictionary()
        total_penalty = 0

        # this function accepts the array, not the dictionary

        if self.params_to_constrain is not None:
            param_penalties = self._param_penalties(lens_args_to_vary_array)
            total_penalty += param_penalties

        lens_args_to_vary = self.param_class.argstovary_todictionary(lens_args_to_vary_array)
        self.lens_args_latest = lens_args_to_vary+params_fixed

        img_penalty = self._image_position_penalty(lens_args_to_vary, self._chi_mode)

        total_penalty += img_penalty

        centroid_penalty = self._centroid_penalty(lens_args_to_vary)
        total_penalty += centroid_penalty

        self._compute_mags = self._compute_mags_criterion()

        if self._compute_mags:
            mag_penalty = self._magnification_penalty(lens_args_to_vary)
            total_penalty += mag_penalty

        self._book_keeping(img_penalty, centroid_penalty, mag_penalty, param_penalties)

        self._counter += 1

        return total_penalty

    def _reset(self, compute_mags=False):

        self.mag_penalty, self.src_penalty, self.param_penalty, self.parameters, \
        self.centroid_penalty = [], [], [], [], []

        self._counter = 1
        self._compute_mags = compute_mags
        self.is_converged = False

    def _get_total_pen(self):

        return np.array(self.src_penalty) + np.array(self.mag_penalty) + np.array(self.centroid_penalty) + \
                np.array(self.param_penalty)

    def _get_best(self):

        total = self._get_total_pen()

        index = np.argmin(total)

        self.src_pen_best = self.src_penalty[index]

        return total[index]

    def _book_keeping(self,src_penalty,centroid_penalty,mag_penalty,param_pen):

        if self._counter % 500 == 0 and self.verbose:
            index = np.argmin(self._get_total_pen())
            print('source penalty: ', self.src_penalty[index])
            print('centroid penalty: ', self.centroid_penalty[index])

            if mag_penalty is not None:
                print('mag penalty: ', self.mag_penalty[index])
            if self.params_to_constrain is not None:
                print('param penalty: ',self.param_penalty[index])

        if mag_penalty is None:
            mag_penalty = 10**10
        if src_penalty is None:
            src_penalty = 10**10
        if centroid_penalty is None:
            centroid_penalty = 10**10
        if param_pen is None:
            param_pen = 10**10

        self.src_penalty.append(src_penalty)
        self.mag_penalty.append(mag_penalty)
        self.centroid_penalty.append(centroid_penalty)
        self.param_penalty.append(param_pen)

        self.parameters.append(self.lens_args_latest)
        self._test_convergence()
        self._compute_mags_criterion()

    def _init_particles(self,n_particles,n_iterations):

        self._n_total_iter = n_iterations*n_particles
        self._n_particles = n_particles

    def _compute_mags_criterion(self):

        if self.tol_mag is None:
            return False

        if self._compute_mags:
            return True

        if self._counter <= self._n_particles:
            return False

        if min(self.src_penalty[-self._n_particles:]) < self._pso_compute_magnification:
            return True
        else:
            return False

    def _test_convergence(self):

        if self._counter <= self._n_particles:
            self.is_converged = False
            return

        if min(self.src_penalty[-self._n_particles:]) < self._pso_convergence_mean:
            self.is_converged = True
        else:
            self.is_converged = False

    def _param_penalties(self,lens_args_tovary):

        penalty = 0

        for pname in self.params_to_constrain.keys():

            if pname == 'shear' and hasattr(self.param_class.routine, '_fixshear'):
                continue

            if pname == 'shear':

                index1 = self.param_class.routine.params_to_vary.index('shear_e1')
                index2 = self.param_class.routine.params_to_vary.index('shear_e2')
                shear,_ = cart2polar(lens_args_tovary[index1],lens_args_tovary[index2])

                penalty += 0.5 * ((shear - self.params_to_constrain['shear'][0])*self.params_to_constrain['shear'][1]**-1)**2

            elif pname == 'shear_pa':

                index1 = self.param_class.routine.params_to_vary.index('shear_e1')
                index2 = self.param_class.routine.params_to_vary.index('shear_e2')
                _, shear_pa = cart2polar(lens_args_tovary[index1], lens_args_tovary[index2])

                penalty += 0.5 * (
                            (shear_pa - self.params_to_constrain['shear_pa'][0]) * self.params_to_constrain['shear_pa'][1] ** -1) ** 2

            else:

                index = self.param_class.routine.params_to_vary.index(pname)
                value = lens_args_tovary[index]
                target = self.params_to_constrain[pname][0]
                sigma = self.params_to_constrain[pname][1]

                penalty += 0.5*((value-target)*sigma**-1)**2

        return penalty

    def _image_position_penalty(self, lens_args_tovary, chi_mode):

        self.betax,self.betay = self.lensing._ray_shooting_fast(lens_args_tovary)

        dx_source = ((self.betax[0] - self.betax[1]) ** 2 + (self.betax[0] - self.betax[2]) ** 2 + (
                    self.betax[0] - self.betax[3]) ** 2 + (
                      self.betax[1] - self.betax[2]) ** 2 +
              (self.betax[1] - self.betax[3]) ** 2 + (self.betax[2] - self.betax[3]) ** 2)
        dy_source = ((self.betay[0] - self.betay[1]) ** 2 + (self.betay[0] - self.betay[2]) ** 2 + (
                    self.betay[0] - self.betay[3]) ** 2 + (
                      self.betay[1] - self.betay[2]) ** 2 +
              (self.betay[1] - self.betay[3]) ** 2 + (self.betay[2] - self.betay[3]) ** 2)

        src_plane_pen = 0.5 * (dx_source + dy_source) * self.tol_source ** -2

        if chi_mode == 0:
            return src_plane_pen
        elif src_plane_pen > self._pso_convergence_mean:
            return src_plane_pen

        else:

            # compute the positions in the image plane
            kwargs_lens_final = lens_args_tovary + self.param_class.argsfixed_todictionary()
            x_image, y_image = self._solver.findBrightImage(np.mean(self.betax), np.mean(self.betay),
                                                            kwargs_lens_final, arrival_time_sort=False)

            # if we have the wrong number of images, use the source plane chi^2
            if len(x_image) != len(self._ximage) or len(y_image) != len(self._yimage):
                return src_plane_pen

            # compute the image plane chi^2
            else:
                # try to match image positions, compute chi^2
                inds = sort_image_index(x_image, y_image, self._ximage, self._yimage)

                dx = (x_image[inds] - self._ximage) ** 2
                dy = (y_image[inds] - self._yimage) ** 2
                image_plane_penalty = 0
                if isinstance(self.tol_image, list) or isinstance(self.tol_image, np.ndarray):
                    for i, dx_dy in enumerate(self.tol_image):
                        image_plane_penalty += 0.5 * (dx[i] + dy[i]) * self.tol_image[i] ** -2
                else:
                    image_plane_penalty = np.sum(0.5 * (dx + dy) * self.tol_image ** -2)

                return image_plane_penalty

    def _magnification_penalty(self,lens_args):

        magnifications = self.lensing._magnification_fast(lens_args)

        magnifications *= max(magnifications) ** -1

        self._mags = magnifications

        dM = []

        for i, target in enumerate(self.magnification_target):
            if target != 0:
                mag_tol = self.tol_mag[i] * target
                dM.append((magnifications[i] - target) * mag_tol ** -1)

        dM = np.array(dM)

        return 0.5 * np.sum(dM ** 2)

    def _centroid_penalty(self, values_dic):

        dx = (values_dic[0]['center_x'] - self.centroid_0[0])*self.tol_centroid**-1
        dy = (values_dic[0]['center_y'] - self.centroid_0[1])*self.tol_centroid**-1

        return 0.5*(dx**2+dy**2)