import numpy as np
import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util
from scipy.optimize import minimize


class LensEquationSolver(object):
    """
    class to solve for image positions given lens model and source position
    """
    def __init__(self, lensModel):
        """

        :param lensModel: instance of a class according to lenstronomy.LensModel.lens_model
        This class must contain the following definitions (with same syntax as the standard LensModel() class:
        def ray_shooting()
        def hessian()
        def magnification()
        """
        self.lensModel = lensModel

    def _static_lens_settings(self, kwargs_lens):
        """

        :param kwargs_lens: lens model keyword argument list
        :return: LensModel() instance without observed lensed positions, kwargs_lens with transformed positions into
        physical space
        """
        if self.lensModel.multi_plane is True:
            kwargs_lens = self.lensModel.lens_model.observed2physical_convention(kwargs_lens)
            self.lensModel.lens_model.ignore_observed_positions = True
        return kwargs_lens

    def _make_dynamic(self):
        """
        undo ignored observational position settings

        :return:
        """
        if self.lensModel.multi_plane is True:
            self.lensModel.lens_model.ignore_observed_positions = False

    def image_position_stochastic(self, source_x, source_y, kwargs_lens, search_window=10,
                                  precision_limit=10**(-10), arrival_time_sort=True, x_center=0,
                                  y_center=0, num_random=1000):
        """
        Solves the lens equation stochastically with the scipy minimization routine on the quadratic distance between
        the backwards ray-shooted proposed image position and the source position.
        Credits to Giulia Pagano

        :param source_x: source position
        :param source_y: source position
        :param kwargs_lens: lens model list of keyword arguments
        :param search_window: angular size of search window
        :param precision_limit: limit required on the precision in the source plane
        :param arrival_time_sort: bool, if True sorts according to arrival time
        :param x_center: center of search window
        :param y_center: center of search window
        :param num_random: number of random starting points of the non-linear solver in the search window
        :param verbose: bool, if True, prints performance information
        :return: x_image, y_image
        """
        kwargs_lens = self._static_lens_settings(kwargs_lens)

        x_solve, y_solve = [], []
        for i in range(num_random):
            x_init = np.random.uniform(-search_window / 2., search_window / 2) + x_center
            y_init = np.random.uniform(-search_window / 2., search_window / 2) + y_center
            xinitial = np.array([x_init, y_init])
            result = minimize(self._root, xinitial, args=(kwargs_lens, source_x, source_y), tol=precision_limit ** 2, method='Nelder-Mead')
            if self._root(result.x, kwargs_lens, source_x, source_y) < precision_limit**2:
                x_solve.append(result.x[0])
                y_solve.append(result.x[1])

        x_mins, y_mins = image_util.findOverlap(x_solve, y_solve, precision_limit)
        if arrival_time_sort is True:
            x_mins, y_mins = self.sort_arrival_times(x_mins, y_mins, kwargs_lens)
        self._make_dynamic()
        return x_mins, y_mins

    def _root(self, x, kwargs_lens, source_x, source_y):
        """

        :param x: parameters [x-coord, y-coord]
        :param kwargs_lens: list of keyword arguments of the lens model
        :param source_x: source position
        :param source_y: source position
        :return: square distance between ray-traced image position and given source position
        """
        x_, y_ = x
        beta_x, beta_y = self.lensModel.ray_shooting(x_, y_, kwargs_lens)
        return (beta_x - source_x)**2 + (beta_y - source_y)**2

    def image_position_from_source(self, sourcePos_x, sourcePos_y, kwargs_lens, min_distance=0.1, search_window=10,
                                   precision_limit=10**(-10), num_iter_max=100, arrival_time_sort=True,
                                   initial_guess_cut=True, verbose=False, x_center=0, y_center=0, num_random=0,
                                   non_linear=False):
        """
        finds image position source position and lense model

        :param sourcePos_x: source position in units of angle
        :param sourcePos_y: source position in units of angle
        :param kwargs_lens: lens model parameters as keyword arguments
        :param min_distance: minimum separation to consider for two images in units of angle
        :param search_window: window size to be considered by the solver. Will not find image position outside this window
        :param precision_limit: required precision in the lens equation solver (in units of angle in the source plane).
        :param num_iter_max: maximum iteration of lens-source mapping conducted by solver to match the required precision
        :param arrival_time_sort: bool, if True, sorts image position in arrival time (first arrival photon first listed)
        :param initial_guess_cut: bool, if True, cuts initial local minima selected by the grid search based on distance criteria from the source position
        :param verbose: bool, if True, prints some useful information for the user
        :param x_center: float, center of the window to search for point sources
        :param y_center: float, center of the window to search for point sources
        :param non_linear: bool, if True applies a non-linear solver not dependent on Hessian computation
        :returns: (exact) angular position of (multiple) images ra_pos, dec_pos in units of angle
        :raises: AttributeError, KeyError
        """
        kwargs_lens = self._static_lens_settings(kwargs_lens)

        # compute number of pixels to cover the search window with the required min_distance
        numPix = int(round(search_window / min_distance) + 0.5)
        x_grid, y_grid = util.make_grid(numPix, min_distance)
        x_grid += x_center
        y_grid += y_center
        # ray-shoot to find the relative distance to the required source position for each grid point
        x_mapped, y_mapped = self.lensModel.ray_shooting(x_grid, y_grid, kwargs_lens)
        absmapped = util.displaceAbs(x_mapped, y_mapped, sourcePos_x, sourcePos_y)
        # select minima in the grid points and select grid points that do not deviate more than the
        # width of the grid point to a solution of the lens equation
        x_mins, y_mins, delta_map = util.neighborSelect(absmapped, x_grid, y_grid)
        if verbose is True:
            print("There are %s regions identified that could contain a solution of the lens equation" % len(x_mins))
        #mag = np.abs(mag)
        #print(x_mins, y_mins, 'before requirement of min_distance')
        if initial_guess_cut is True:
            mag = np.abs(self.lensModel.magnification(x_mins, y_mins, kwargs_lens))
            mag[mag < 1] = 1
            x_mins = x_mins[delta_map <= min_distance*mag*5]
            y_mins = y_mins[delta_map <= min_distance*mag*5]
            if verbose is True:
                print("The number of regions that meet the plausibility criteria are %s" % len(x_mins))
        x_mins = np.append(x_mins, np.random.uniform(low=-search_window/2+x_center, high=search_window/2+x_center,
                                                     size=num_random))
        y_mins = np.append(y_mins, np.random.uniform(low=-search_window / 2 + y_center, high=search_window / 2 + y_center,
                                             size=num_random))
        # iterative solving of the lens equation for the selected grid points
        x_mins, y_mins, solver_precision = self._findIterative(x_mins, y_mins, sourcePos_x, sourcePos_y, kwargs_lens,
                                                               precision_limit, num_iter_max, verbose=verbose,
                                                               min_distance=min_distance, non_linear=non_linear)
        # only select iterative results that match the precision limit
        x_mins = x_mins[solver_precision <= precision_limit]
        y_mins = y_mins[solver_precision <= precision_limit]
        # find redundant solutions within the min_distance criterion
        x_mins, y_mins = image_util.findOverlap(x_mins, y_mins, min_distance)
        if arrival_time_sort is True:
            x_mins, y_mins = self.sort_arrival_times(x_mins, y_mins, kwargs_lens)
        self._make_dynamic()
        return x_mins, y_mins

    def _findIterative(self, x_min, y_min, sourcePos_x, sourcePos_y, kwargs_lens, precision_limit=10 ** (-10),
                       num_iter_max=100, verbose=False, min_distance=0.01, non_linear=False):
        num_candidates = len(x_min)
        x_mins = np.zeros(num_candidates)
        y_mins = np.zeros(num_candidates)
        solver_precision = np.zeros(num_candidates)
        for i in range(len(x_min)):
            x_guess, y_guess, delta, l = self._solve_single_proposal(x_min[i], y_min[i], sourcePos_x, sourcePos_y,
                                                                  kwargs_lens, precision_limit, num_iter_max,
                                                                     min_distance, non_linear=non_linear)
            if verbose is True:
                print("Solution found for region %s with required precision at iteration %s" % (i, l))
            x_mins[i] = x_guess
            y_mins[i] = y_guess
            solver_precision[i] = delta
        return x_mins, y_mins, solver_precision

    def _solve_single_proposal(self, x_guess, y_guess, source_x, source_y, kwargs_lens, precision_limit, num_iter_max,
                               max_step, non_linear=False):
        l = 0
        if non_linear is True:
        #if self.lensModel.multi_plane is True:
            xinitial = np.array([x_guess, y_guess])
            result = minimize(self._root, xinitial, args=(kwargs_lens, source_x, source_y), tol=precision_limit ** 2,
                              method='Nelder-Mead')
            delta = self._root(result.x, kwargs_lens, source_x, source_y)
            x_guess, y_guess = result.x[0], result.x[1]

        else:
            x_mapped, y_mapped = self.lensModel.ray_shooting(x_guess, y_guess, kwargs_lens)
            delta = np.sqrt((x_mapped - source_x) ** 2 + (y_mapped - source_y) ** 2)

            while (delta > precision_limit and l < num_iter_max):
                x_mapped, y_mapped = self.lensModel.ray_shooting(x_guess, y_guess, kwargs_lens)
                delta = np.sqrt((x_mapped - source_x) ** 2 + (y_mapped - source_y) ** 2)
                f_xx, f_xy, f_yx, f_yy = self.lensModel.hessian(x_guess, y_guess, kwargs_lens)
                DistMatrix = np.array([[1 - f_yy, f_yx], [f_xy, 1 - f_xx]])
                det = (1 - f_xx) * (1 - f_yy) - f_xy * f_yx
                deltaVec = np.array([x_mapped - source_x, y_mapped - source_y])
                image_plane_vector = DistMatrix.dot(deltaVec) / det
                dist = np.sqrt(image_plane_vector[0]**2 + image_plane_vector[1]**2)
                if dist > max_step:
                    image_plane_vector *= max_step/dist
                x_guess, y_guess, delta, l = self._do_step(x_guess, y_guess, source_x, source_y, delta,
                                                                  image_plane_vector, kwargs_lens, l, num_iter_max)
        return x_guess, y_guess, delta, l

    def _do_step(self, x_guess, y_guess, source_x, source_y, delta_init, image_plane_vector, kwargs_lens, iter_num,
                 num_iter_max):
        x_new = x_guess - image_plane_vector[0]
        y_new = y_guess - image_plane_vector[1]
        x_mapped, y_mapped = self.lensModel.ray_shooting(x_new, y_new, kwargs_lens)
        delta_new = np.sqrt((x_mapped - source_x) ** 2 + (y_mapped - source_y) ** 2)
        iter_num += 1
        if delta_new > delta_init:
            if num_iter_max < iter_num:
                return x_guess, y_guess, delta_init, iter_num
            else:
                #
                #image_plane_vector /= 10
                image_plane_vector[0] *= np.random.normal(loc=0, scale=0.5)
                image_plane_vector[1] *= np.random.normal(loc=0, scale=0.5)
                return self._do_step(x_guess, y_guess, source_x, source_y, delta_init, image_plane_vector, kwargs_lens, iter_num, num_iter_max)
        else:
            return x_new, y_new, delta_new, iter_num

    def findBrightImage(self, sourcePos_x, sourcePos_y, kwargs_lens, numImages=4, min_distance=0.01, search_window=5,
                        precision_limit=10**(-10), num_iter_max=10, arrival_time_sort=True):
        """

        :param sourcePos_x:
        :param sourcePos_y:
        :param deltapix:
        :param numPix:
        :param magThresh: magnification threshold for images to be selected
        :param numImage: number of selected images (will select the highest magnified ones)
        :param kwargs_lens:
        :param ray_shooting_function: a special function for performing ray shooting; defaults to self.lensModel.ray_shooting
        :param hessian_function: same as ray_shooting_function, but for computing the hessian matrix
        :param magnification_function: same as ray_shooting_function, but for computing magnifications
        :return:
        """

        x_mins, y_mins = self.image_position_from_source(sourcePos_x, sourcePos_y, kwargs_lens, min_distance,
                                                         search_window, precision_limit, num_iter_max,
                                                         arrival_time_sort=arrival_time_sort)
        mag_list = []
        for i in range(len(x_mins)):
            mag = self.lensModel.magnification(x_mins[i], y_mins[i], kwargs_lens)
            mag_list.append(abs(mag))
        mag_list = np.array(mag_list)
        x_mins_sorted = util.selectBest(x_mins, mag_list, numImages)
        y_mins_sorted = util.selectBest(y_mins, mag_list, numImages)
        return x_mins_sorted, y_mins_sorted

    def sort_arrival_times(self, x_mins, y_mins, kwargs_lens):
        """
        sort arrival times (fermat potential) of image positions in increasing order of light travel time
        :param x_mins: ra position of images
        :param y_mins: dec position of images
        :param kwargs_lens: keyword arguments of lens model
        :return: sorted lists of x_mins and y_mins
        """

        if hasattr(self.lensModel, '_no_potential'):
            raise Exception('Instance of lensModel passed to this class does not compute the lensing potential, '
                            'and therefore cannot compute time delays.')

        if len(x_mins) <= 1:
            return x_mins, y_mins
        x_source, y_source = self.lensModel.ray_shooting(x_mins, y_mins, kwargs_lens)
        x_source = np.mean(x_source)
        y_source = np.mean(y_source)
        if self.lensModel.multi_plane is True:
            arrival_time = self.lensModel.arrival_time(x_mins, y_mins, kwargs_lens)
        else:
            fermat_pot = self.lensModel.fermat_potential(x_mins, y_mins, x_source, y_source, kwargs_lens)
            arrival_time = fermat_pot
        idx = np.argsort(arrival_time)
        x_mins = np.array(x_mins)[idx]
        y_mins = np.array(y_mins)[idx]
        return x_mins, y_mins
