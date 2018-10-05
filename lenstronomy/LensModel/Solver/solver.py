from lenstronomy.LensModel.Solver.solver2point import Solver2Point
from lenstronomy.LensModel.Solver.solver4point import Solver4Point
import numpy as np


class Solver(object):
    """
    joint solve class to manage with type of solver to be executed and checks whether the requirements are fulfilled.

    """

    def __init__(self, solver_type, lensModel, num_images=0):
        """

        :param solver_type: string, option for specific solver type
        see detailed instruction of the Solver4Point and Solver2Point classes
        :param lensModel: instance of a LensModel() class
        :param num_images: int, number of images to be solved for
        """
        self._num_images = num_images
        self._lensModel = lensModel
        if self._num_images == 4:
            self._solver = Solver4Point(lensModel, solver_type=solver_type)
        elif self. _num_images == 2:
            self._solver = Solver2Point(lensModel, solver_type=solver_type)
        else:
            raise ValueError("%s number of images is not valid. Use 2 or 4!" % self._num_images)

    def constraint_lensmodel(self, x_pos, y_pos, kwargs_list, xtol=1.49012e-12):
        """

        :param x_pos:
        :param y_pos:
        :param kwargs_list:
        :return:
        """
        return self._solver.constraint_lensmodel(x_pos, y_pos, kwargs_list, xtol=xtol)

    def update_solver(self, kwargs_lens, kwargs_ps):
        x_, y_ = kwargs_ps[0]['ra_image'], kwargs_ps[0]['dec_image']
        if not len(x_) == self._num_images:
            raise ValueError("Point source number %s must be as specified by the solver with number of images %s" %
                             (len(x_), self._num_images))
        kwargs_lens, precision = self.constraint_lensmodel(x_, y_, kwargs_lens)
        return kwargs_lens

    def check_solver(self, kwargs_lens, kwargs_ps):
        """
        returns the precision of the solver to match the image position

        :param kwargs_lens: full lens model (including solved parameters)
        :param kwargs_ps: point source keyword arguments
        :return: precision of Euclidean distances between the different rays arriving at the image positions
        """
        ra_image, dec_image = kwargs_ps[0]['ra_image'], kwargs_ps[0]['dec_image']
        source_x, source_y = self._lensModel.ray_shooting(ra_image, dec_image,
                                                         kwargs_lens)
        dist = np.sqrt((source_x - source_x[0]) ** 2 + (source_y - source_y[0]) ** 2)
        return dist

    def add_fixed_lens(self, kwargs_fixed_lens, kwargs_lens_init):
        """
        returns kwargs that are kept fixed during run, depending on options
        :param kwargs_options:
        :param kwargs_lens:
        :return:
        """
        kwargs_fixed_lens = self._solver.add_fixed_lens(kwargs_fixed_lens, kwargs_lens_init)
        return kwargs_fixed_lens
