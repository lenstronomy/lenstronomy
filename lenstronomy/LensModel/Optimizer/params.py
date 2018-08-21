from lenstronomy.LensModel.Optimizer.fixed_routines import *
import numpy as np


class Params(object):

    known_routines = ['fixed_powerlaw_shear']

    def __init__(self, zlist=None, lens_list=None, arg_list=None,
                 optimizer_routine=str, xpos = None, ypos = None):

        assert optimizer_routine in self.known_routines

        self.routine_name = optimizer_routine

        if optimizer_routine == 'fixed_powerlaw_shear':
            routine = FixedPowerLaw_Shear(lens_list,arg_list,xpos,ypos)

        else:
            raise ValueError("optimizer_routine must be called %s" %self.known_routines)

        self._zlist = zlist
        self._lens_list = lens_list
        self._arg_list = arg_list

        self._k_start = routine._k_start
        self._Ntovary = routine._Ntovary
        self.tovary_indicies = routine._tovary_indicies
        self.routine = routine

        self.zlist_tovary,self.lenslist_tovary,self.args_tovary = self.to_vary()
        self.zlist_fixed, self.lenslist_fixed, self.args_fixed = self.fixed()

    def to_vary_limits(self,re_optimize):

        lower_limit,upper_limit = self.routine.get_param_ranges(re_optimize)

        return lower_limit,upper_limit

    def to_vary(self):
        zlist_tovary = self._zlist[0:self._Ntovary]
        lenslist_tovary = self._lens_list[0:self._Ntovary]
        args_tovary = self._arg_list[0:self._Ntovary]

        return zlist_tovary, lenslist_tovary, args_tovary

    def tovary_array(self):

        values = []

        for index in self._tovary_indicies:
            for name in self.routine.to_vary_names[index]:
                values.append(self.args_tovary[index][name])
        return np.array(values)

    def fixed(self):
        zlist_fixed = self._zlist[self._Ntovary:]
        lenslist_fixed = self._lens_list[self._Ntovary:]
        args_fixed = self._arg_list[self._Ntovary:]

        return zlist_fixed, lenslist_fixed, args_fixed

    def argsfixed_values(self):

        return np.array([args.values() for args in self.args_fixed])

    def argstovary_todictionary(self,values):

        args_list = []
        count = 0

        for n in range(0, int(self._Ntovary)):

            args = {}

            for key in self.routine.param_names[n]:

                if key in self.routine.fixed_names[n]:
                    args.update({key:self.routine.fixed_values[n][key]})
                else:
                    args.update({key:values[count]})
                    count += 1

            args_list.append(args)

        return args_list

    def argsfixed_todictionary(self):

        return self.args_fixed

    def _kwargs_to_tovary(self,kwargs):

        values = []

        for n in range(0, int(self._Ntovary)):

            for key in self.routine.param_names[n]:

                if key not in self.routine.fixed_names[n]:
                    values.append(kwargs[n][key])

        return np.array(values)




