from astropy.cosmology import FlatLambdaCDM, FlatwCDM, LambdaCDM, w0waCDM


class CosmoParam(object):
    """
    class for managing the parameters involved
    """
    def __init__(self, cosmology, kwargs_lower, kwargs_upper, kwargs_fixed, ppn_sampling=False):
        """

        :param cosmology: string describing cosmological model
        :param ppn_sampling: post-newtonian parameter sampling
        :param kwargs_lower: keyword arguments with lower limits of parameters
        :param kwargs_upper: keyword arguments with upper limits of parameters
        :param kwargs_fixed: keyword arguments and values of fixed parameters
        """
        self._cosmology = cosmology
        self._ppn_sampling = ppn_sampling
        if cosmology not in ['FLCDM', "FwCDM", "w0waCDM", "oLCDM"]:
            raise ValueError('cosmology %s not supported!' % cosmology)
        self._kwargs_fixed = kwargs_fixed
        self._kwargs_lower = kwargs_lower
        self._kwargs_upper = kwargs_upper

    @property
    def num_param(self):
        """

        :return: number of free parameters and list of their names in order
        """
        num = 0
        list = []
        if 'h0' not in self._kwargs_fixed:
            num += 1
            list.append('h0')
        if self._cosmology in ['FLCDM', "FwCDM", "w0waCDM", "oLCDM"]:
            if 'om' not in self._kwargs_fixed:
                num += 1
                list.append('om')
        if self._cosmology in ["FwCDM"]:
            if 'w' not in self._kwargs_fixed:
                num += 1
                list.append('w')
        if self._cosmology in ["w0waCDM"]:
            if 'w0' not in self._kwargs_fixed:
                num += 1
                list.append('w0')
            if 'wa' not in self._kwargs_fixed:
                num += 1
                list.append('wa')
        if self._cosmology in ["oLCDM"]:
            if 'ok' not in self._kwargs_fixed:
                num += 1
                list.append('ok')
        if self._ppn_sampling is True:
            if 'gamma_ppn' not in self._kwargs_fixed:
                num += 1
                list.append('gamma_ppn')
        return num, list

    def args2kwargs(self, args):
        """

        :param args: sampling argument list
        :return: keyword argument list with parameter names
        """
        i = 0
        kwargs = {}
        if 'h0' in self._kwargs_fixed:
            kwargs['h0'] = self._kwargs_fixed['h0']
        else:
            kwargs['h0'] = args[i]
            i += 1
        if self._cosmology in ['FLCDM', "FwCDM", "w0waCDM", "oLCDM"]:
            if 'om' in self._kwargs_fixed:
                kwargs['om'] = self._kwargs_fixed['om']
            else:
                kwargs['om'] = args[i]
                i += 1
        if self._cosmology in ["FwCDM"]:
            if 'w' in self._kwargs_fixed:
                kwargs['w'] = self._kwargs_fixed['w']
            else:
                kwargs['w'] = args[i]
                i += 1
        if self._cosmology in ["w0waCDM"]:
            if 'w0' in self._kwargs_fixed:
                kwargs['w0'] = self._kwargs_fixed['w0']
            else:
                kwargs['w0'] = args[i]
                i += 1
            if 'wa' in self._kwargs_fixed:
                kwargs['wa'] = self._kwargs_fixed['wa']
            else:
                kwargs['wa'] = args[i]
                i += 1
        if self._cosmology in ["oLCDM"]:
            if 'ok' in self._kwargs_fixed:
                kwargs['ok'] = self._kwargs_fixed['ok']
            else:
                kwargs['ok'] = args[i]
                i += 1
        if self._ppn_sampling is True:
            if 'gamma_ppn' in self._kwargs_fixed:
                kwargs['gamma_ppn'] = self._kwargs_fixed['gamma_ppn']
            else:
                kwargs['gamma_ppn'] = args[i]
                i += 1
        return kwargs

    def kwargs2args(self, kwargs):
        """

        :param kwargs: keyword argument list of parameters
        :return: sampling argument list in specified order
        """
        args = []
        if 'h0' not in self._kwargs_fixed:
            args.append(kwargs['h0'])
        if self._cosmology in ['FLCDM', "FwCDM", "w0waCDM", "oLCDM"]:
            if 'om' not in self._kwargs_fixed:
                args.append(kwargs['om'])
        if self._cosmology in ["FwCDM"]:
            if 'w' not in self._kwargs_fixed:
                args.append(kwargs['w'])
        if self._cosmology in ["w0waCDM"]:
            if 'w0' not in self._kwargs_fixed:
                args.append(kwargs['w0'])
            if 'wa' not in self._kwargs_fixed:
                args.append(kwargs['wa'])

        if self._cosmology in ["oLCDM"]:
            if 'ok' not in self._kwargs_fixed:
                args.append(kwargs['ok'])
        if self._ppn_sampling is True:
            if 'gamma_ppn' not in self._kwargs_fixed:
                args.append(kwargs['gamma_ppn'])
        return args

    def cosmo(self, kwargs):
        """

        :param kwargs: keyword arguments of parameters
        :return: astropy.cosmology instance
        """
        if self._cosmology == "FLCDM":
            cosmo = FlatLambdaCDM(H0=kwargs['h0'], Om0=kwargs['om'])
        elif self._cosmology == "FwCDM":
            cosmo = FlatwCDM(H0=kwargs['h0'], Om0=kwargs['om'], w0=kwargs['w'])
        elif self._cosmology == "w0waCDM":
            cosmo = w0waCDM(H0=kwargs['h0'], Om0=kwargs['om'], Ode0=1.0 - kwargs['om'], w0=kwargs['w0'], wa=kwargs['wa'])
        elif self._cosmology == "oLCDM":
            cosmo = LambdaCDM(H0=kwargs['h0'], Om0=kwargs['om'], Ode0=1.0 - kwargs['om'] - kwargs['ok'])
        else:
            raise ValueError("I don't know the cosmology %s" % self._cosmology)
        return cosmo

    @property
    def param_bounds(self):
        """

        :return: argument list of the hard bounds
        """
        lowerlimit = self.kwargs2args(self._kwargs_lower)
        upperlimit = self.kwargs2args(self._kwargs_upper)
        return lowerlimit, upperlimit
