import numpy as np
import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util


class LensEquationSolver(object):
    """
    class to solve for image positions given lens model and source position
    """
    def __init__(self, lensModel):
        """

        :param imsim: imsim class
        """
        self.lensModel = lensModel

    def image_position_from_source(self, sourcePos_x, sourcePos_y, kwargs_lens, min_distance=0.1, search_window=10,
                                   precision_limit=10**(-10), num_iter_max=100, arrival_time_sort=True):
        """
        finds image position source position and lense model

        :param sourcePos_x: source position in units of angle
        :param sourcePos_y: source position in units of angle
        :param kwargs_lens: lens model parameters as keyword arguments
        :param min_distance: minimum separation to consider for two images in units of angle
        :param search_window: window size to be considered by the solver. Will not find image position outside this window
        :param precision_limit: required precision in the lens equation solver (in units of angle in the source plane).
        :param num_iter_max: maximum iteration of lens-source mapping conducted by solver to match the required precision
        :returns:  (exact) angular position of (multiple) images ra_pos, dec_pos in units of angle
        :raises: AttributeError, KeyError
        """
        # compute number of pixels to cover the search window with the required min_distance
        numPix = int(round(search_window / min_distance) + 0.5)
        x_grid, y_grid = util.make_grid(numPix, min_distance)
        # ray-shoot to find the relative distance to the required source position for each grid point
        x_mapped, y_mapped = self.lensModel.ray_shooting(x_grid, y_grid, kwargs_lens)
        absmapped = util.displaceAbs(x_mapped, y_mapped, sourcePos_x, sourcePos_y)
        # select minima in the grid points and select grid points that do not deviate more than the
        # width of the grid point to a solution of the lens equation
        x_mins, y_mins, delta_map = util.neighborSelect(absmapped, x_grid, y_grid)
        #mag = self.lensModel.magnification(x_mins, y_mins, kwargs_lens)
        #mag = np.abs(mag)
        #print(x_mins, y_mins, 'before requirement of min_distance')
        #x_mins = x_mins[delta_map*mag <= min_distance*5]
        #y_mins = y_mins[delta_map*mag <= min_distance*5]
        #print(x_mins, y_mins, 'after requirement of min_distance')
        # iterative solving of the lens equation for the selected grid points
        x_mins, y_mins, solver_precision = self._findIterative(x_mins, y_mins, sourcePos_x, sourcePos_y, kwargs_lens, precision_limit, num_iter_max)
        # only select iterative results that match the precision limit
        x_mins = x_mins[solver_precision <= precision_limit]
        y_mins = y_mins[solver_precision <= precision_limit]
        #print(x_mins, y_mins, 'after precision limit requirement')
        # find redundant solutions within the min_distance criterion
        x_mins, y_mins = image_util.findOverlap(x_mins, y_mins, min_distance)
        #print(x_mins, y_mins, 'after overlap removals')
        if arrival_time_sort is True:
            x_mins, y_mins = self.sort_arrival_times(x_mins, y_mins, kwargs_lens)
        #x_mins, y_mins = lenstronomy_util.coordInImage(x_mins, y_mins, numPix, deltapix)
        return x_mins, y_mins

    def _findIterative(self, x_min, y_min, sourcePos_x, sourcePos_y, kwargs_lens, precision_limit=10**(-10), num_iter_max=100):
        """
        find iterative solution to the demanded level of precision for the pre-selected regions given a lense model and source position

        :param mins: indices of local minimas found with def neighborSelect and def valueSelect
        :type mins: 1d numpy array
        :returns:  (n,3) numpy array with exact position, displacement and magnification [posAngel,delta,mag]
        :raises: AttributeError, KeyError
        """
        num_candidates = len(x_min)
        x_mins = np.zeros(num_candidates)
        y_mins = np.zeros(num_candidates)
        solver_precision = np.zeros(num_candidates)
        for i in range(len(x_min)):
            l = 0
            x_mapped, y_mapped = self.lensModel.ray_shooting(x_min[i], y_min[i], kwargs_lens)
            delta = np.sqrt((x_mapped - sourcePos_x)**2+(y_mapped - sourcePos_y)**2)
            f_xx, f_xy, f_yx, f_yy = self.lensModel.hessian(x_min[i], y_min[i], kwargs_lens)
            DistMatrix = np.array([[1 - f_yy, f_yx], [f_xy, 1 - f_xx]])
            det = (1 - f_xx) * (1 - f_yy) - f_xy * f_yx
            posAngel = np.array([x_min[i], y_min[i]])
            while(delta > precision_limit and l < num_iter_max):
                deltaVec = np.array([x_mapped - sourcePos_x, y_mapped - sourcePos_y])
                posAngel = posAngel - DistMatrix.dot(deltaVec)/det
                x_mapped, y_mapped = self.lensModel.ray_shooting(posAngel[0], posAngel[1], kwargs_lens)
                delta = np.sqrt((x_mapped - sourcePos_x)**2+(y_mapped - sourcePos_y)**2)
                f_xx, f_xy, f_yx, f_yy = self.lensModel.hessian(posAngel[0], posAngel[1], kwargs_lens)
                DistMatrix = np.array([[1 - f_yy, f_xy], [f_yx, 1 - f_xx]])
                det = (1 - f_xx) * (1 - f_yy) - f_xy * f_xy
                l += 1
            x_mins[i] = posAngel[0]
            y_mins[i] = posAngel[1]
            solver_precision[i] = delta
        return x_mins, y_mins, solver_precision

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
        if len(x_mins) <= 1:
            return x_mins, y_mins
        x_source, y_source = self.lensModel.ray_shooting(x_mins, y_mins, kwargs_lens)
        x_source = np.mean(x_source)
        y_source = np.mean(y_source)
        if self.lensModel.multi_plane:
            arrival_time = self.lensModel.lens_model.arrival_time(x_mins, y_mins, kwargs_lens)
        else:
            fermat_pot = self.lensModel.fermat_potential(x_mins, y_mins, x_source, y_source, kwargs_lens)
            arrival_time = -fermat_pot
        idx = np.argsort(arrival_time)
        x_mins = np.array(x_mins)[idx]
        y_mins = np.array(y_mins)[idx]
        return x_mins, y_mins