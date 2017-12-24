import numpy as np
import astrofunc.util as util
import lenstronomy.Util.image_util as lenstronomy_util
from lenstronomy.LensModel.lens_model import LensModel


class LensEquationSolver(object):
    """
    class to solve for image positions given lens model and source position
    """
    def __init__(self, lens_model_list, foreground_shear=False):
        """

        :param imsim: imsim class
        """
        self.LensModel = LensModel(lens_model_list, foreground_shear)

    def image_position_from_source(self, sourcePos_x, sourcePos_y, kwargs_lens, kwargs_else=None, min_distance=0.01, search_window=5, precision_limit=10**(-10), num_iter_max=100):
        """
        finds image position source position and lense model

        :param sourcePos_x: source position in units of angle
        :param sourcePos_y: source position in units of angle
        :param kwargs_lens: lens model parameters as keyword arguments
        :param kwargs_else: other parameters as keyword arguments
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
        x_mapped, y_mapped = self.LensModel.ray_shooting(x_grid, y_grid, kwargs_lens, kwargs_else)
        absmapped = util.displaceAbs(x_mapped, y_mapped, sourcePos_x, sourcePos_y)
        # select minima in the grid points and select grid points that do not deviate more than the
        # width of the grid point to a solution of the lens equation
        x_mins, y_mins, delta_map = util.neighborSelect(absmapped, x_grid, y_grid)
        x_mins = x_mins[delta_map <= min_distance]
        y_mins = y_mins[delta_map <= min_distance]
        # iterative solving of the lens equation for the selected grid points
        x_mins, y_mins, solver_precision = self._findIterative(x_mins, y_mins, sourcePos_x, sourcePos_y, kwargs_lens, kwargs_else, precision_limit, num_iter_max)
        # only select iterative results that match the precision limit
        x_mins = x_mins[solver_precision <= precision_limit]
        y_mins = y_mins[solver_precision <= precision_limit]
        # find redundant solutions within the min_distance criterion
        x_mins, y_mins = lenstronomy_util.findOverlap(x_mins, y_mins, min_distance)
        x_mins, y_mins = self.sort_arrival_times(x_mins, y_mins, kwargs_lens, kwargs_else)
        #x_mins, y_mins = lenstronomy_util.coordInImage(x_mins, y_mins, numPix, deltapix)

        return x_mins, y_mins

    def _findIterative(self, x_min, y_min, sourcePos_x, sourcePos_y, kwargs_lens, kwargs_else=None, precision_limit=10**(-10), num_iter_max=100):
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
            x_mapped, y_mapped = self.LensModel.ray_shooting(x_min[i], y_min[i], kwargs_lens, kwargs_else)
            delta = np.sqrt((x_mapped - sourcePos_x)**2+(y_mapped - sourcePos_y)**2)
            f_xx, f_xy, f_yy = self.LensModel.hessian(x_min[i], y_min[i], kwargs_lens, kwargs_else)
            DistMatrix = np.array([[1 - f_yy, f_xy], [f_xy, 1 - f_xx]])
            det = (1 - f_xx) * (1 - f_yy) - f_xy * f_xy
            posAngel = np.array([x_min[i], y_min[i]])
            while(delta > precision_limit and l < num_iter_max):
                deltaVec = np.array([x_mapped - sourcePos_x, y_mapped - sourcePos_y])
                posAngel = posAngel - DistMatrix.dot(deltaVec)/det
                x_mapped, y_mapped = self.LensModel.ray_shooting(posAngel[0], posAngel[1], kwargs_lens, kwargs_else)
                delta = np.sqrt((x_mapped - sourcePos_x)**2+(y_mapped - sourcePos_y)**2)
                f_xx, f_xy, f_yy = self.LensModel.hessian(posAngel[0], posAngel[1], kwargs_lens, kwargs_else)
                DistMatrix = np.array([[1 - f_yy, f_xy], [f_xy, 1 - f_xx]])
                det = (1 - f_xx) * (1 - f_yy) - f_xy * f_xy
                l += 1
            x_mins[i] = posAngel[0]
            y_mins[i] = posAngel[1]
            solver_precision[i] = delta
        return x_mins, y_mins, solver_precision

    def findBrightImage(self, sourcePos_x, sourcePos_y, kwargs_lens, kwargs_else=None, numImages=4, min_distance=0.01, search_window=5, precision_limit=10**(-10), num_iter_max=10):
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
        x_mins, y_mins = self.image_position_from_source(sourcePos_x, sourcePos_y, kwargs_lens, kwargs_else, min_distance, search_window, precision_limit, num_iter_max)
        mag_list = []
        for i in range(len(x_mins)):
            mag = self.LensModel.magnification(x_mins[i], y_mins[i], kwargs_lens, kwargs_else)
            mag_list.append(abs(mag))
        mag_list = np.array(mag_list)
        x_mins_sorted = util.selectBest(x_mins, mag_list, numImages)
        y_mins_sorted = util.selectBest(y_mins, mag_list, numImages)
        return x_mins_sorted, y_mins_sorted

    def sort_arrival_times(self, x_mins, y_mins, kwargs_lens, kwargs_else=None):
        """
        sort arrival times (fermat potential) of image positions
        :param x_mins:
        :param y_mins:
        :param kwargs_lens:
        :param kwargs_else:
        :return:
        """
        x_source, y_source = self.LensModel.ray_shooting(x_mins, y_mins, kwargs_lens, kwargs_else)
        x_source = np.mean(x_source)
        y_source = np.mean(y_source)
        fermat_pot = self.LensModel.fermat_potential(x_mins, y_mins, x_source, y_source, kwargs_lens, kwargs_else)
        idx = np.argsort(fermat_pot)
        x_mins = np.array(x_mins)[idx]
        y_mins = np.array(y_mins)[idx]
        return x_mins, y_mins