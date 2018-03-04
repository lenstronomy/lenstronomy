from lenstronomy.LensModel.Solver.solver2point import Solver2Point
from lenstronomy.LensModel.Solver.solver4point import Solver4Point


class Solver(object):
    """
    joint solve class to manage with type of solver to be executed and checks whether the requirements are fulfilled.

    """

    def __init__(self, solver_type, lensModel=None, num_images=0):
        """

        :param solver_type:
        :param lensModel:
        :param num_images:
        """
        self._solver_type = solver_type
        self._lensModel = lensModel
        self._num_images = num_images

        if self._num_images == 4:
            self.solver = Solver4Point(self._lensModel, solver_type=self._solver_type)
        elif self. _num_images == 2:
            self.solver = Solver2Point(self._lensModel, solver_type=self._solver_type)
        else:
            raise ValueError("%s number of images is not valid. Use 2 or 4!" % self._num_images)

    def constraint_lensmodel(self, x_pos, y_pos, kwargs_list):
        """

        :param x_pos:
        :param y_pos:
        :param kwargs_list:
        :return:
        """
        return self.solver.constraint_lensmodel(x_pos, y_pos, kwargs_list)

    def update_solver(self, kwargs_lens, kwargs_ps):
        x_, y_ = kwargs_ps[0]['ra_image'], kwargs_ps[0]['dec_image']
        if not len(x_) == self._num_images:
            raise ValueError("Point source number %s must be as specified by the solver with number of images %s" %
                             (len(x_), self._num_images))
        kwargs_lens, precision = self.constraint_lensmodel(x_, y_, kwargs_lens)
        return kwargs_lens

    def add_fixed_lens(self, kwargs_fixed_lens, kwargs_lens_init):
        """
        returns kwargs that are kept fixed during run, depending on options
        :param kwargs_options:
        :param kwargs_lens:
        :return:
        """
        kwargs_fixed_lens = self.solver.add_fixed_lens(kwargs_fixed_lens, kwargs_lens_init)
        return kwargs_fixed_lens
