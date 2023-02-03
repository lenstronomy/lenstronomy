__author__ = 'mgomer'

from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
import numpy as np
import copy
from lenstronomy.Util import param_util
from lenstronomy.Util import util
from lenstronomy.LensModel.lens_model import LensModel



class SynthesisProfile(LensProfileBase):
    """
    Similar to CSEProductAvgSet, this class is more general. Describes a linear sum of many simple profiles.

    Example (default): Mimic an NFW profile with many CSE profiles. If given the right weights, similar to NFW_ELLIPSE_CSE class.
    Can instead use LinearWeightFit class to fit for the right weights instead of importing them.
    """
    profile_name = 'SYNTHESIS'
    def __init__(self, target_lens_model, component_lens_model, kwargs_list, lin_fit_hyperparams,product_average=True):
        """
        kwargs_list: the normalization (must be nonzero) will be effectively overridden by the linear weights
        product_average: if True, indicates that the class profile provided is evaluated at r=sqrt(q)*r_major_axis
        """

        self.target_class = LensModel([target_lens_model])
        self.component_class = LensModel([component_lens_model])
        self.kwargs_list=kwargs_list
        self.lin_fit_hyperparams=lin_fit_hyperparams

    def LinearWeightMLEFit(self, kwargs_target, kwargs_list):

        self.set_limits(kwargs_list, self.lin_fit_hyperparams)
        Y = self.target_class.kappa(x=self.r_eval_list, y=np.zeros_like(self.r_eval_list), kwargs=kwargs_target)
        M = np.zeros((len(self.r_eval_list),self.num_components))
        C = np.diag(Y * self.sigma) #covariance matrix between components
        for j in range(self.num_components): # M[i,j] is the jth component 1D kappa evaluated at r[i].
            kwargs = self.circular_centered_kwargs(kwargs_list[j])
            M[:, j] = self.component_class.kappa(x=self.r_eval_list, y=0, kwargs=[kwargs])
        MTinvC=np.matmul(M.T,np.linalg.inv(C))
        first_term=np.linalg.inv(np.matmul(MTinvC,M))
        second_term=np.matmul(MTinvC,Y)
        return np.matmul(first_term,second_term)

    def circular_centered_kwargs(self,kwargs):

        kwargs_new=copy.deepcopy(kwargs)
        if 'e1' in kwargs_new:
            kwargs_new['e1']=0
        if 'e2' in kwargs_new:
            kwargs_new['e2']=0
        if 'center_x' in kwargs_new:
            kwargs_new['center_x']=0
        if 'center_y' in kwargs_new:
            kwargs_new['center_y']=0
        return kwargs_new


    def set_limits(self, kwargs_list, lin_fit_hyperparams):

        self.num_components=len(kwargs_list)
        if 'lower_log_bound' not in lin_fit_hyperparams:
            self.lower_log_bound = -6
        else:
            self.lower_log_bound = lin_fit_hyperparams['lower_log_bound']
        if 'upper_log_bound' not in lin_fit_hyperparams:
            self.upper_log_bound = 3
        else:
            self.upper_log_bound = lin_fit_hyperparams['upper_log_bound']
        if 'num_r_evals' not in lin_fit_hyperparams:
                self.num_r_evals = 100
        else:
            self.num_r_evals = lin_fit_hyperparams['num_r_evals']
        if 'sigma' not in lin_fit_hyperparams:
            self.sigma = 0.01
        else:
            self.sigma = lin_fit_hyperparams['sigma']
        self.r_eval_list=np.logspace(self.lower_log_bound,self.upper_log_bound,self.num_r_evals)

    def function(self, x, y, **kwargs_target):
        """


        """
        # TODO: how to handle offset centers? Need to be at zero for fitting, then place at center_x, center_y
        # phi_q, q = param_util.ellipticity2phi_q(e1, e2)
        # # shift
        # x_ = x - center_x
        # y_ = y - center_y
        # # rotate
        # x__, y__ = util.rotate(x_, y_, phi_q)
        weight_list = self.LinearWeightMLEFit([kwargs_target], self.kwargs_list)
        # potential calculation
        f_ = np.zeros_like(x)
        f_innermost = 0 #for some profiles, minimum potential can go below zero. Add a constant here to make zero the minimum
        for kwargs, weight in zip(self.kwargs_list, weight_list):
            f_ += weight * self.component_class.potential(x, y, [kwargs])#-potential_offset)
            f_innermost+=weight * self.component_class.potential([10**self.lower_log_bound], [0], [kwargs])
        return f_ - f_innermost

    def derivatives(self, x, y, **kwargs_target):
        """


        """
        # TODO: how to handle offset centers? Need to be at zero for fitting, then place at center_x, center_y
        # phi_q, q = param_util.ellipticity2phi_q(e1, e2)
        # # shift
        # x_ = x - center_x
        # y_ = y - center_y
        # # rotate
        # x__, y__ = util.rotate(x_, y_, phi_q)

        # potential calculation
        weight_list = self.LinearWeightMLEFit([kwargs_target],self.kwargs_list)
        f_x, f_y = np.zeros_like(x), np.zeros_like(y)
        for kwargs, weight in zip(self.kwargs_list, weight_list):
            f_x_, f_y_ = weight * np.array(self.component_class.alpha(x, y, [kwargs]))
            f_x += f_x_
            f_y += f_y_
        return f_x, f_y

    def hessian(self, x, y, **kwargs_target):
        """


        """
        # TODO: how to handle offset centers? Need to be at zero for fitting, then place at center_x, center_y
        # phi_q, q = param_util.ellipticity2phi_q(e1, e2)
        # # shift
        # x_ = x - center_x
        # y_ = y - center_y
        # # rotate
        # x__, y__ = util.rotate(x_, y_, phi_q)

        # potential calculation
        weight_list = self.LinearWeightMLEFit([kwargs_target], self.kwargs_list)
        f_xx, f_xy, f_yx, f_yy = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        for kwargs, weight in zip(self.kwargs_list, weight_list):
            f_xx_i, f_xy_i, f_yx_i, f_yy_i = weight * np.array(self.component_class.hessian(x, y, [kwargs]))
            f_xx += f_xx_i
            f_xy += f_xy_i
            f_yx += f_yx_i
            f_yy += f_yy_i
        return f_xx, f_xy, f_yx, f_yy


#TODO implement a test of accuracy and a warning if things have gone wrong (for some choices of kwargs_target, can overfit)

