

class CosmoParam(object):
    """
    class that handles the cosmology relevant parameters
    """

    def __init__(self, cosmo_type=None, kwargs_fixed={}):
        """


        :param sampling: bool, if True, activates time-delay parameters
        :param D_dt_init: initial guess of time-delay distance (Mpc)
        :param D_dt_sigma: initial uncertainty
        :param D_dt_lower: lower bound
        :param D_dt_upper: upper bound
        """
        if cosmo_type is None:
            self._sampling = False
        elif cosmo_type == 'D_dt':
            self._sampling = True
        else:
            raise ValueError("cosmo_type %s is not supported!" % cosmo_type)
        self._cosmo_type = cosmo_type
        self._kwargs_fixed = kwargs_fixed

    def getParams(self, args, i):
        """

        :param args:
        :param i:
        :return:
        """
        kwargs_cosmo = {}
        if self._sampling is True:
            if self._cosmo_type == 'D_dt':
                kwargs_cosmo['D_dt'] = args[i]
                i += 1
        return kwargs_cosmo, i

    def setParams(self, kwargs_cosmo):
        """

        :param kwargs:
        :return:
        """
        args = []
        if self._sampling is True:
            if self._cosmo_type == 'D_dt':
                args.append(kwargs_cosmo['D_dt'])
        return args

    def param_init(self, kwargs_mean):
        """

        :param kwargs_mean:
        :return:
        """
        mean = []
        sigma = []
        if self._sampling is True:
            if self._cosmo_type == 'D_dt':
                mean.append(kwargs_mean['D_dt'])
                sigma.append(kwargs_mean['D_dt_sigma'])
        return mean, sigma

    def num_param(self):
        """

        :return:
        """
        num = 0
        list = []
        if self._sampling is True:
            if self._cosmo_type == 'D_dt':
                num += 1
                list.append('D_dt')
        return num, list
