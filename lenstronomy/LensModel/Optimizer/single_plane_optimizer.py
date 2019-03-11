import numpy as np


class SinglePlaneLensing(object):

    _no_potential = True

    def __init__(self, lensmodel, x_pos, y_pos, params, arg_list):

        """
        This class performs (fast) lensing computations for single plane scenarios
        :param lensmodel:
        :param x_pos:
        :param y_pos:
        :param params:
        :param arg_list:
        """

        self._params = params
        self._lensModel = lensmodel
        self._x_pos, self._y_pos = np.array(x_pos), np.array(y_pos)

        self._all_lensmodel_args = arg_list
        k_start = params._k_start

        # compute the foreground deflections and second derivatives from subhalos
        if k_start > 0 and len(arg_list)>k_start:

            self._k_sub = np.arange(k_start, len(arg_list))
            self._k_macro = np.arange(0, k_start)

            # subhalo deflections
            self._alpha_x_sub, self._alpha_y_sub = self._lensModel.alpha(x_pos, y_pos, arg_list, self._k_sub)

            self._sub_fxx, self._sub_fxy, _, self._sub_fyy = self._lensModel.hessian(self._x_pos, self._y_pos,
                                                                                     self._all_lensmodel_args,
                                                                                     self._k_sub)

        else:
            self._k_macro, self._k_sub = None, None
            self._alpha_x_sub, self._alpha_y_sub = 0, 0
            self._sub_fxx, self._sub_fyy, self._sub_fxy = 0, 0, 0

    def _ray_shooting_fast(self, lens_args):

        # compute the macromodel deflection
        alphax_macro,alphay_macro = self._lensModel.alpha(self._x_pos, self._y_pos, lens_args, k=self._k_macro)

        # compute the source position
        betax = self._x_pos - alphax_macro - self._alpha_x_sub
        betay = self._y_pos - alphay_macro - self._alpha_y_sub

        return betax,betay

    def _hessian_fast(self, args):

        fxx_macro, fxy_macro, _, fyy_macro = self._lensModel.hessian(self._x_pos, self._y_pos, args, k=self._k_macro)

        fxx = fxx_macro + self._sub_fxx
        fyy = fyy_macro + self._sub_fyy
        fxy = fxy_macro + self._sub_fxy

        return fxx, fxy, fxy, fyy

    def _magnification_fast(self, args):

        fxx, fxy, _, fyy = self._hessian_fast(args)

        det_J = (1-fxx)*(1-fyy) - fxy**2

        magnifications = np.absolute(det_J**-1)

        magnifications *= max(magnifications)**-1

        return magnifications
