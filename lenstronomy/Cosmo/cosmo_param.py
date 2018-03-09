import numpy as np


class CosmoParam(object):
    """
    class that handles the cosmology relevant parameters
    """

    def __init__(self, sampling=False, D_dt_init=1000, D_dt_sigma=100, D_dt_lower=0, D_dt_upper=10000):
        """


        :param sampling: bool, if True, activates time-delay parameters
        :param D_dt_init: initial guess of time-delay distance (Mpc)
        :param D_dt_sigma: initial uncertainty
        :param D_dt_lower: lower bound
        :param D_dt_upper: upper bound
        """
        self._sampling = sampling
        self._Dt_init = D_dt_init
        self._D_dt_sigma = D_dt_sigma
        self._D_dt_lower = D_dt_lower
        self._D_dt_upper = D_dt_upper

    def getParams(self, args, i):
        """

        :param args:
        :param i:
        :return:
        """
        kwargs_cosmo = {}
        if self._sampling is True:
            kwargs_cosmo['D_dt'] = args[i]
            i += 1
        else:
            kwargs_cosmo['D_dt'] = self._Dt_init
        return kwargs_cosmo, i

    def setParams(self, kwargs_cosmo):
        """

        :param kwargs:
        :return:
        """
        args = []
        if self._sampling is True:
            args.append(kwargs_cosmo['D_dt'])
        return args

    def param_init(self):
        """

        :param kwargs_mean:
        :return:
        """
        mean = []
        sigma = []
        if self._sampling is True:
            mean.append(self._Dt_init)
            sigma.append(self._D_dt_sigma)
        return mean, sigma

    def num_param(self):
        """

        :return:
        """
        num = 0
        list = []
        if self._sampling is True:
            num += 1
            list.append('D_dt')
        return num, list
