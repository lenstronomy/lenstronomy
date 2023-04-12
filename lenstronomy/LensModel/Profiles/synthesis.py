__author__ = 'mgomer'

from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
import numpy as np
import copy
from lenstronomy.Util import param_util
from lenstronomy.Util import util
from lenstronomy.LensModel.lens_model import LensModel



class SynthesisProfile(LensProfileBase):
    """
    A general class which describes a linear sum of many simple profiles to approximate a target profile

    Example: Mimic an NFW profile with many CSE profiles. In this case, you could use LensModel(['SYNTHESIS'],kwargs_synthesis=kwargs_synthesis) with
    kwargs_synthesis={'target_lens_model': 'NFW',
                    'component_lens_model': 'CSE',
                   'kwargs_list': kwargs_list,
                   'lin_fit_hyperparams':{'lower_log_bound':-6, 'upper_log_bound':3, 'num_r_evals':100, 'sigma':0.01} (default values)
                   }
    where kwargs_list would be a list of input CSE kwargs (where the amplitude will be re-adjusted).

    """
    profile_name = 'SYNTHESIS'
    def __init__(self, target_lens_model, component_lens_model, kwargs_list, lin_fit_hyperparams):
        """
        :param target_lens_model: name of target profile
        :param component_lens_model: name of component profile
        :param kwargs_list: list of kwargs of component profile, length of list corresponds to number of components used to fit.
                            The normalization (must be nonzero) will be effectively overridden by the linear weights
        :param lin_fit_hyperparams: kwargs indicating range of fit, number of points to evaluate fit, etc.
        """
        super(SynthesisProfile, self).__init__()
        self.target_class = LensModel([target_lens_model])
        self.component_class = LensModel([component_lens_model])
        self.kwargs_list=kwargs_list
        self.lin_fit_hyperparams=lin_fit_hyperparams
        self.check_num_evals()


    def LinearWeightMLEFit(self, kwargs_target, kwargs_list):
        if self._static is True:
            return self._linear_weights
        else:
            return self._LinearWeightMLEFit(kwargs_target, kwargs_list)

    def _LinearWeightMLEFit(self, kwargs_target, kwargs_list):
        """
        Fits a linear fit of the amplitudes for each component to minimize a chi2.

        :param kwargs_target: kwargs of target profile to be approximated
        :param kwargs_list: list of kwargs of component profile, length of list corresponds to number of components used to fit.
                            The normalization (must be nonzero) will be effectively overridden by the linear weights
        """
        self.set_limits(kwargs_list, self.lin_fit_hyperparams)
        kwargs_target_centered=[self.circular_centered_kwargs(kwargs_target[0])]
        Y = self.target_class.kappa(x=self.r_eval_list, y=np.zeros_like(self.r_eval_list), kwargs=kwargs_target_centered)
        M = np.zeros((len(self.r_eval_list),self.num_components))
        C = np.diag(Y * self.sigma) #covariance matrix between components
        for j in range(self.num_components): # M[i,j] is the jth component 1D kappa evaluated at r[i].
            kwargs = self.circular_centered_kwargs(kwargs_list[j])
            M[:, j] = self.component_class.kappa(x=self.r_eval_list, y=0, kwargs=[kwargs])
        MTinvC=np.matmul(M.T,np.linalg.inv(C))
        first_term=np.linalg.inv(np.matmul(MTinvC,M))
        second_term=np.matmul(MTinvC,Y)
        return np.matmul(first_term,second_term)

    def set_static(self, linear_weights):
        """
        Sets weights to be static self values. Useful to call e.g. function many times with the same kwargs.
        If kwargs_target or kwargs_list change, need to rerun linear fit by using set_dynamic.

        :param linear_weights: output of LinearWeightMLEFit
        :return: self weights set
        """
        self._static = True
        self._linear_weights = linear_weights

    def set_dynamic(self):
        self._static = False
        if hasattr(self, '_linear_weights'):
            del self._linear_weights

    def circular_centered_kwargs(self,kwargs):
        """
        :param kwargs: kwargs to remove center and ellipticity for linear fit. These are re-added when functions are called
        """
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
        """
        :param kwargs_list: list of kwargs of component profile
        :param lin_fit_hyperparams: kwargs indicating range of fit, number of points to evaluate fit, etc.
            'lower_log_bound': log10 innermost radius of fit
            'upper_log_bound': log10 outermost radius of fit
            'num_r_evals': number of locations to evaluate fit to minimize chi2, must be larger than the number of components
            'sigma': used to evaluate chi2. default is 1%
        """
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
        returns lensing potential

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :kwargs_target: kwargs of target profile to be approximated
        """
        weight_list = self.LinearWeightMLEFit([kwargs_target], self.kwargs_list)
        f_ = np.zeros_like(x)
        f_innermost = 0 #for some profiles, minimum potential can go below zero. Add a constant here to make zero the minimum
        for kwargs, weight in zip(self.kwargs_list, weight_list):
            f_ += weight * self.component_class.potential(x, y, [kwargs])#-potential_offset)
            f_innermost+=weight * self.component_class.potential([10**self.lower_log_bound], [0], [kwargs])
        return f_ - f_innermost

    def derivatives(self, x, y, **kwargs_target):
        """
        returns df/dx and df/dy of the function which are the deflection angles

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :kwargs_target: kwargs of target profile to be approximated
        """
        weight_list = self.LinearWeightMLEFit([kwargs_target],self.kwargs_list)
        f_x, f_y = np.zeros_like(x), np.zeros_like(y)
        for kwargs, weight in zip(self.kwargs_list, weight_list):
            f_x_, f_y_ = weight * np.array(self.component_class.alpha(x, y, [kwargs]))
            f_x += f_x_
            f_y += f_y_
        return f_x, f_y

    def hessian(self, x, y, **kwargs_target):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :kwargs_target: kwargs of target profile to be approximated
        """
        weight_list = self.LinearWeightMLEFit([kwargs_target], self.kwargs_list)
        f_xx, f_xy, f_yx, f_yy = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        for kwargs, weight in zip(self.kwargs_list, weight_list):
            f_xx_i, f_xy_i, f_yx_i, f_yy_i = weight * np.array(self.component_class.hessian(x, y, [kwargs]))
            f_xx += f_xx_i
            f_xy += f_xy_i
            f_yx += f_yx_i
            f_yy += f_yy_i
        return f_xx, f_xy, f_yx, f_yy

    def check_num_evals(self):
        """
        Confirm that the number of evaluations is more than the number of components. Still not guaranteed to prevent overfitting
        """
        num_comp=len(self.kwargs_list)
        if num_comp >= self.lin_fit_hyperparams['num_r_evals']:
            raise ValueError('There must be more num_r_evals than components or the profile will be overfit')

