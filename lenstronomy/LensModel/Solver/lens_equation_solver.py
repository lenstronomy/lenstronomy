import numpy as np
import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util
from scipy.optimize import minimize
from lenstronomy.LensModel.Solver.epl_shear_solver import solve_lenseq_pemd

__all__ = ["LensEquationSolver"]


class LensEquationSolver(object):
    """Class to solve for image positions given lens model and source position."""

    def __init__(self, lensModel):
        """This class must contain the following definitions (with same syntax as the
        standard LensModel() class: def ray_shooting() def hessian() def magnification()

        :param lensModel: instance of a class according to
            lenstronomy.LensModel.lens_model
        """
        self.lensModel = lensModel

    def image_position_stochastic(
        self,
        source_x,
        source_y,
        kwargs_lens,
        search_window=10,
        precision_limit=10 ** (-10),
        arrival_time_sort=True,
        x_center=0,
        y_center=0,
        num_random=1000,
    ):
        """Solves the lens equation stochastic with the scipy minimization routine on
        the quadratic distance between the backwards ray-shooted proposed image position
        and the source position. Credits to Giulia Pagano.

        :param source_x: source position
        :param source_y: source position
        :param kwargs_lens: lens model list of keyword arguments
        :param search_window: angular size of search window
        :param precision_limit: limit required on the precision in the source plane
        :param arrival_time_sort: bool, if True sorts according to arrival time
        :param x_center: center of search window
        :param y_center: center of search window
        :param num_random: number of random starting points of the non-linear solver in
            the search window
        :return: x_image, y_image
        """
        kwargs_lens = self.lensModel.set_static(kwargs_lens)

        x_solve, y_solve = [], []
        for i in range(num_random):
            x_init = (
                np.random.uniform(-search_window / 2.0, search_window / 2) + x_center
            )
            y_init = (
                np.random.uniform(-search_window / 2.0, search_window / 2) + y_center
            )
            xinitial = np.array([x_init, y_init])
            result = minimize(
                self._root,
                xinitial,
                args=(kwargs_lens, source_x, source_y),
                tol=precision_limit**2,
                method="Nelder-Mead",
            )
            if (
                self._root(result.x, kwargs_lens, source_x, source_y)
                < precision_limit**2
            ):
                x_solve.append(result.x[0])
                y_solve.append(result.x[1])

        x_mins, y_mins = image_util.findOverlap(x_solve, y_solve, precision_limit)
        if arrival_time_sort:
            x_mins, y_mins = self.sort_arrival_times(x_mins, y_mins, kwargs_lens)
        self.lensModel.set_dynamic()
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
        return (beta_x - source_x) ** 2 + (beta_y - source_y) ** 2

    def candidate_solutions(
        self,
        sourcePos_x,
        sourcePos_y,
        kwargs_lens,
        min_distance=0.1,
        search_window=10,
        verbose=False,
        x_center=0,
        y_center=0,
    ):
        """Finds pixels in the image plane possibly hosting a solution of the lens
        equation, for the given source position and lens model.

        :param sourcePos_x: source position in units of angle
        :param sourcePos_y: source position in units of angle
        :param kwargs_lens: lens model parameters as keyword arguments
        :param min_distance: minimum separation to consider for two images in units of
            angle
        :param search_window: window size to be considered by the solver. Will not find
            image position outside this window
        :param verbose: bool, if True, prints some useful information for the user
        :param x_center: float, center of the window to search for point sources
        :param y_center: float, center of the window to search for point sources
        :returns: (approximate) angular position of (multiple) images ra_pos, dec_pos in
            units of angles, related ray-traced source displacements and pixel width
        :raises: AttributeError, KeyError
        """
        kwargs_lens = self.lensModel.set_static(kwargs_lens)
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
        x_mins, y_mins, delta_map = util.local_minima_2d(absmapped, x_grid, y_grid)
        # pixel width
        pixel_width = x_grid[1] - x_grid[0]

        return x_mins, y_mins, delta_map, pixel_width

    def image_position_analytical(
        self,
        x,
        y,
        kwargs_lens,
        arrival_time_sort=True,
        magnification_limit=None,
        **kwargs_solver,
    ):
        """Solves the lens equation. Only supports EPL-like (plus shear) models. Uses a
        specialized recipe that solves a one-dimensional lens equation that is easier
        and more reliable to solve than the usual two-dimensional lens equation.

        :param x: source position in units of angle, an array of positions is also supported.
        :param y: source position in units of angle, an array of positions is also supported.
        :param kwargs_lens: lens model parameters as keyword arguments
        :param arrival_time_sort: bool, if True, sorts image position in arrival time (first arrival photon first listed)
        :param magnification_limit: None or float, if set will only return image positions that have an
         abs(magnification) larger than this number
        :param kwargs_solver: additional kwargs to be supplied to the solver. Particularly relevant are Nmeas and Nmeas_extra
        :returns: (exact) angular position of (multiple) images ra_pos, dec_pos in units of angle
         Note: in contrast to the other solvers, generally the (heavily demagnified) central image will also be included, so
         setting a a proper magnification_limit is more important. To get similar behaviour, a limit of 1e-1 is acceptable
        """
        lens_model_list = list(self.lensModel.lens_model_list)
        if lens_model_list not in (
            ["SIE", "SHEAR"],
            ["SIE"],
            ["EPL_NUMBA", "SHEAR"],
            ["EPL_NUMBA"],
            ["EPL", "SHEAR"],
            ["EPL"],
        ):
            raise ValueError(
                "Only SIE, EPL, EPL_NUMBA (+shear) supported in the analytical solver for now."
            )

        x_mins, y_mins = solve_lenseq_pemd((x, y), kwargs_lens, **kwargs_solver)
        if arrival_time_sort:
            x_mins, y_mins = self.sort_arrival_times(x_mins, y_mins, kwargs_lens)
        if magnification_limit is not None:
            mag = np.abs(self.lensModel.magnification(x_mins, y_mins, kwargs_lens))
            x_mins = x_mins[mag >= magnification_limit]
            y_mins = y_mins[mag >= magnification_limit]
        return x_mins, y_mins

    def image_position_from_source(
        self, sourcePos_x, sourcePos_y, kwargs_lens, solver="lenstronomy", **kwargs
    ):
        """Solves the lens equation, i.e. finds the image positions in the lens plane
        that are mapped to a given source position.

        :param sourcePos_x: source position in units of angle
        :param sourcePos_y: source position in units of angle
        :param kwargs_lens: lens model parameters as keyword arguments
        :param solver: which solver to use, can be 'lenstronomy' (default), 'analytical'
            or 'stochastic'.
        :param kwargs: Any additional kwargs are passed to the chosen solver, see the
            documentation of image_position_lenstronomy, image_position_analytical and
            image_position_stochastic
        :returns: (exact) angular position of (multiple) images ra_pos, dec_pos in units
            of angle
        """
        if solver == "lenstronomy":
            return self.image_position_lenstronomy(
                sourcePos_x, sourcePos_y, kwargs_lens, **kwargs
            )
        if solver == "analytical":
            return self.image_position_analytical(
                sourcePos_x, sourcePos_y, kwargs_lens, **kwargs
            )
        if solver == "stochastic":
            return self.image_position_stochastic(
                sourcePos_x, sourcePos_y, kwargs_lens, **kwargs
            )
        raise ValueError(f"{solver} is not a valid solver.")

    def image_position_lenstronomy(
        self,
        sourcePos_x,
        sourcePos_y,
        kwargs_lens,
        min_distance=0.1,
        search_window=10,
        precision_limit=10 ** (-10),
        num_iter_max=100,
        arrival_time_sort=True,
        initial_guess_cut=True,
        verbose=False,
        x_center=0,
        y_center=0,
        num_random=0,
        non_linear=False,
        magnification_limit=None,
    ):
        """Finds image position  given source position and lens model. The solver first
        samples does a grid search in the lens plane, and the grid points that are
        closest to the supplied source position are fed to a specialized gradient-based
        root finder that finds the exact solutions. Works with all lens models.

        :param sourcePos_x: source position in units of angle
        :param sourcePos_y: source position in units of angle
        :param kwargs_lens: lens model parameters as keyword arguments
        :param min_distance: minimum separation to consider for two images in units of
            angle
        :param search_window: window size to be considered by the solver. Will not find
            image position outside this window
        :param precision_limit: required precision in the lens equation solver (in units
            of angle in the source plane).
        :param num_iter_max: maximum iteration of lens-source mapping conducted by
            solver to match the required precision
        :param arrival_time_sort: bool, if True, sorts image position in arrival time
            (first arrival photon first listed)
        :param initial_guess_cut: bool, if True, cuts initial local minima selected by
            the grid search based on distance criteria from the source position
        :param verbose: bool, if True, prints some useful information for the user
        :param x_center: float, center of the window to search for point sources
        :param y_center: float, center of the window to search for point sources
        :param num_random: int, number of random positions within the search window to
            be added to be starting positions for the gradient decent solver
        :param non_linear: bool, if True applies a non-linear solver not dependent on
            Hessian computation
        :param magnification_limit: None or float, if set will only return image
            positions that have an abs(magnification) larger than this number
        :returns: (exact) angular position of (multiple) images ra_pos, dec_pos in units
            of angle
        :raises: AttributeError, KeyError
        """
        # find pixels in the image plane possibly hosting a solution of the lens equation, related source distances and
        # pixel width
        x_mins, y_mins, delta_map, pixel_width = self.candidate_solutions(
            sourcePos_x,
            sourcePos_y,
            kwargs_lens,
            min_distance,
            search_window,
            verbose,
            x_center,
            y_center,
        )
        if verbose:
            print(
                "There are %s regions identified that could contain a solution of the lens equation with"
                "coordinates %s and %s " % (len(x_mins), x_mins, y_mins)
            )
        if len(x_mins) < 1:
            return x_mins, y_mins
        if initial_guess_cut:
            mag = np.abs(self.lensModel.magnification(x_mins, y_mins, kwargs_lens))
            mag[mag < 1] = 1
            x_mins = x_mins[delta_map <= min_distance * mag * 5]
            y_mins = y_mins[delta_map <= min_distance * mag * 5]
            if verbose:
                print(
                    "The number of regions that meet the plausibility criteria are %s"
                    % len(x_mins)
                )
        x_mins = np.append(
            x_mins,
            np.random.uniform(
                low=-search_window / 2 + x_center,
                high=search_window / 2 + x_center,
                size=num_random,
            ),
        )
        y_mins = np.append(
            y_mins,
            np.random.uniform(
                low=-search_window / 2 + y_center,
                high=search_window / 2 + y_center,
                size=num_random,
            ),
        )
        # iterative solving of the lens equation for the selected grid points
        # print("Candidates:", x_mins.shape, y_mins.shape)
        x_mins, y_mins, solver_precision = self._find_gradient_decent(
            x_mins,
            y_mins,
            sourcePos_x,
            sourcePos_y,
            kwargs_lens,
            precision_limit,
            num_iter_max,
            verbose=verbose,
            min_distance=min_distance,
            non_linear=non_linear,
        )
        # only select iterative results that match the precision limit
        x_mins = x_mins[solver_precision <= precision_limit]
        y_mins = y_mins[solver_precision <= precision_limit]
        # find redundant solutions within the min_distance criterion
        x_mins, y_mins = image_util.findOverlap(x_mins, y_mins, min_distance)
        if arrival_time_sort:
            x_mins, y_mins = self.sort_arrival_times(x_mins, y_mins, kwargs_lens)
        if magnification_limit is not None:
            mag = np.abs(self.lensModel.magnification(x_mins, y_mins, kwargs_lens))
            x_mins = x_mins[mag >= magnification_limit]
            y_mins = y_mins[mag >= magnification_limit]
        self.lensModel.set_dynamic()
        return x_mins, y_mins

    def _find_gradient_decent(
        self,
        x_min,
        y_min,
        sourcePos_x,
        sourcePos_y,
        kwargs_lens,
        precision_limit=10 ** (-10),
        num_iter_max=200,
        verbose=False,
        min_distance=0.01,
        non_linear=False,
    ):
        """Given a 'good guess' of a solution of the lens equation (expected image
        position given a fixed source position) this routine iteratively performs a ray-
        tracing with second order correction (effectively gradient decent) to find a
        precise solution to the lens equation.

        :param x_min: np.array, list of 'good guess' solutions of the lens equation
        :param y_min: np.array, list of 'good guess' solutions of the lens equation
        :param sourcePos_x: source position for which to solve the lens equation
        :param sourcePos_y: source position for which to solve the lens equation
        :param kwargs_lens: keyword argument list of the lens model
        :param precision_limit: float, required match in the solution in the source
            plane
        :param num_iter_max: int, maximum number of iterations before the algorithm
            stops
        :param verbose: bool, if True inserts print statements about the behavior of the
            solver
        :param min_distance: maximum correction applied per step (to avoid over-shooting
            in unstable regions)
        :param non_linear: bool, if True, uses scipy.miminize instead of the directly
            implemented gradient decent approach.
        :return: x_position array, y_position array, error in the source plane array
        """
        num_candidates = len(x_min)
        x_mins = np.zeros(num_candidates)
        y_mins = np.zeros(num_candidates)
        solver_precision = np.zeros(num_candidates)
        for i in range(len(x_min)):
            x_guess, y_guess, delta, l = self._solve_single_proposal(
                x_min[i],
                y_min[i],
                sourcePos_x,
                sourcePos_y,
                kwargs_lens,
                precision_limit,
                num_iter_max,
                max_step=min_distance,
                non_linear=non_linear,
            )
            if verbose:
                print(
                    "Solution found for region %s with required precision at iteration %s"
                    % (i, l)
                )
            x_mins[i] = x_guess
            y_mins[i] = y_guess
            solver_precision[i] = delta
        return x_mins, y_mins, solver_precision

    def _solve_single_proposal(
        self,
        x_guess,
        y_guess,
        source_x,
        source_y,
        kwargs_lens,
        precision_limit,
        num_iter_max,
        max_step,
        non_linear=False,
    ):
        """Gradient decent solution of a single proposed starting point (close to a true
        solution)

        :param x_guess: starting guess position in the image plane
        :param y_guess: starting guess position in the image plane
        :param source_x: source position to solve for in the image plane
        :param source_y: source position to solve for in the image plane
        :param kwargs_lens: keyword argument list of the lens model
        :param precision_limit: float, required match in the solution in the source
            plane
        :param num_iter_max: int, maximum number of iterations before the algorithm
            stops
        :param max_step: maximum correction applied per step (to avoid over-shooting in
            instable regions)
        :param non_linear: bool, if True, uses scipy.miminize instead of the directly
            implemented gradient decent approach.
        :return: x_position, y_position, error in the source plane, steps required (for
            gradient decent)
        """
        l = 0
        if non_linear:
            xinitial = np.array([x_guess, y_guess])
            result = minimize(
                self._root,
                xinitial,
                args=(kwargs_lens, source_x, source_y),
                tol=precision_limit**2,
                method="Nelder-Mead",
            )
            delta = self._root(result.x, kwargs_lens, source_x, source_y)
            x_guess, y_guess = result.x[0], result.x[1]

        else:
            x_mapped, y_mapped = self.lensModel.ray_shooting(
                x_guess, y_guess, kwargs_lens
            )
            delta = np.sqrt((x_mapped - source_x) ** 2 + (y_mapped - source_y) ** 2)

            while delta > precision_limit and l < num_iter_max:
                x_mapped, y_mapped = self.lensModel.ray_shooting(
                    x_guess, y_guess, kwargs_lens
                )
                delta = np.sqrt((x_mapped - source_x) ** 2 + (y_mapped - source_y) ** 2)
                f_xx, f_xy, f_yx, f_yy = self.lensModel.hessian(
                    x_guess, y_guess, kwargs_lens
                )
                DistMatrix = np.array([[1 - f_yy, f_yx], [f_xy, 1 - f_xx]])
                det = (1 - f_xx) * (1 - f_yy) - f_xy * f_yx
                deltaVec = np.array([x_mapped - source_x, y_mapped - source_y])
                image_plane_vector = DistMatrix.dot(deltaVec) / det
                dist = np.sqrt(image_plane_vector[0] ** 2 + image_plane_vector[1] ** 2)
                if dist > max_step:
                    image_plane_vector *= max_step / dist
                x_guess, y_guess, delta, l = self._gradient_step(
                    x_guess,
                    y_guess,
                    source_x,
                    source_y,
                    delta,
                    image_plane_vector,
                    kwargs_lens,
                    l,
                    num_iter_max,
                )
        return x_guess, y_guess, delta, l

    def _gradient_step(
        self,
        x_guess,
        y_guess,
        source_x,
        source_y,
        delta_init,
        image_plane_vector,
        kwargs_lens,
        iter_num,
        num_iter_max,
    ):
        """

        :param x_guess: float, current best fit solution in the image plane
        :param y_guess: float, current best fit solution in the image plane
        :param source_x: float, source position to be matched
        :param source_y: float, source position ot be matched
        :param delta_init: current precision in the source plane of the mapped solution
        :param image_plane_vector: correction vector in the image plane based on the Hessian operator and the deviation
         in the source plane
        :param kwargs_lens: lens model keyword argument list
        :param iter_num: int, current iteration number
        :param num_iter_max: int, maximum iteration number before aborting the process
        :return: updated image position in x, updated image position in y, updated precision in the source plane,
         total iterations done after this call
        """
        x_new = x_guess - image_plane_vector[0]
        y_new = y_guess - image_plane_vector[1]
        x_mapped, y_mapped = self.lensModel.ray_shooting(x_new, y_new, kwargs_lens)
        delta_new = np.sqrt((x_mapped - source_x) ** 2 + (y_mapped - source_y) ** 2)
        iter_num += 1
        if delta_new > delta_init:
            if num_iter_max < iter_num:
                return x_guess, y_guess, delta_init, iter_num
            else:
                # if the new proposal is worse than the previous one, it randomly draws a new proposal in a different
                # direction and tries again
                image_plane_vector[0] *= np.random.normal(loc=0, scale=0.5)
                image_plane_vector[1] *= np.random.normal(loc=0, scale=0.5)
                return self._gradient_step(
                    x_guess,
                    y_guess,
                    source_x,
                    source_y,
                    delta_init,
                    image_plane_vector,
                    kwargs_lens,
                    iter_num,
                    num_iter_max,
                )
        else:
            return x_new, y_new, delta_new, iter_num

    def findBrightImage(
        self,
        sourcePos_x,
        sourcePos_y,
        kwargs_lens,
        numImages=4,
        min_distance=0.01,
        search_window=5,
        precision_limit=10 ** (-10),
        num_iter_max=10,
        arrival_time_sort=True,
        x_center=0,
        y_center=0,
        num_random=0,
        non_linear=False,
        magnification_limit=None,
        initial_guess_cut=True,
        verbose=False,
    ):
        """

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
        :param num_random: int, number of random positions within the search window to be added to be starting
         positions for the gradient decent solver
        :param non_linear: bool, if True applies a non-linear solver not dependent on Hessian computation
        :param magnification_limit: None or float, if set will only return image positions that have an
         abs(magnification) larger than this number
        :returns: (exact) angular position of (multiple) images ra_pos, dec_pos in units of angle
        """

        x_mins, y_mins = self.image_position_from_source(
            sourcePos_x,
            sourcePos_y,
            kwargs_lens,
            min_distance=min_distance,
            search_window=search_window,
            precision_limit=precision_limit,
            num_iter_max=num_iter_max,
            arrival_time_sort=arrival_time_sort,
            initial_guess_cut=initial_guess_cut,
            verbose=verbose,
            x_center=x_center,
            y_center=y_center,
            num_random=num_random,
            non_linear=non_linear,
            magnification_limit=magnification_limit,
        )
        mag_list = []
        for i in range(len(x_mins)):
            mag = self.lensModel.magnification(x_mins[i], y_mins[i], kwargs_lens)
            mag_list.append(abs(mag))
        mag_list = np.array(mag_list)
        x_mins_sorted = util.selectBest(x_mins, mag_list, numImages)
        y_mins_sorted = util.selectBest(y_mins, mag_list, numImages)
        if arrival_time_sort:
            x_mins_sorted, y_mins_sorted = self.sort_arrival_times(
                x_mins_sorted, y_mins_sorted, kwargs_lens
            )
        return x_mins_sorted, y_mins_sorted

    def sort_arrival_times(self, x_mins, y_mins, kwargs_lens):
        """Sort arrival times (fermat potential) of image positions in increasing order
        of light travel time.

        :param x_mins: ra position of images
        :param y_mins: dec position of images
        :param kwargs_lens: keyword arguments of lens model
        :return: sorted lists of x_mins and y_mins
        """

        if hasattr(self.lensModel, "_no_potential"):
            raise Exception(
                "Instance of lensModel passed to this class does not compute the lensing potential, "
                "and therefore cannot compute time delays."
            )

        if len(x_mins) <= 1:
            return x_mins, y_mins
        if self.lensModel.multi_plane:
            arrival_time = self.lensModel.arrival_time(x_mins, y_mins, kwargs_lens)
        else:
            fermat_pot = self.lensModel.fermat_potential(x_mins, y_mins, kwargs_lens)
            arrival_time = fermat_pot
        idx = np.argsort(arrival_time)
        x_mins = np.array(x_mins)[idx]
        y_mins = np.array(y_mins)[idx]
        return x_mins, y_mins
