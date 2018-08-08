from lenstronomy.LensModel.Optimizer.fixed_routines import *
import numpy as np


class Params(object):

    known_routines = ['fixed_powerlaw_shear']

    def __init__(self, zlist=None, lens_list=None, arg_list=None,
                 optimizer_routine=str, xpos = None, ypos = None):

        assert optimizer_routine in self.known_routines

        if optimizer_routine == 'fixed_powerlaw_shear':
            routine = FixedPowerLaw_Shear(lens_list,arg_list,xpos,ypos)

        else:
            raise ValueError("optimizer_routine must be called %s" %self.known_routines)

        self.zlist = zlist
        self.lens_list = lens_list
        self.arg_list = arg_list

        self.k_start = routine.k_start
        self.Ntovary = routine.Ntovary
        self.tovary_indicies = routine.tovary_indicies
        self.routine = routine

        self.zlist_tovary,self.lenslist_tovary,self.args_tovary = self.to_vary()
        self.zlist_fixed, self.lenslist_fixed, self.args_fixed = self.fixed()

    def to_vary_limits(self,re_optimize):

        lower_limit,upper_limit = self.routine.get_param_ranges(re_optimize)

        return lower_limit,upper_limit

    def to_vary(self):
        zlist_tovary = self.zlist[0:self.Ntovary]
        lenslist_tovary = self.lens_list[0:self.Ntovary]
        args_tovary = self.arg_list[0:self.Ntovary]

        return zlist_tovary, lenslist_tovary, args_tovary

    def tovary_array(self):

        values = []

        for index in self.tovary_indicies:
            for name in self.routine.to_vary_names[index]:
                values.append(self.args_tovary[index][name])
        return np.array(values)

    def fixed(self):
        zlist_fixed = self.zlist[self.Ntovary:]
        lenslist_fixed = self.lens_list[self.Ntovary:]
        args_fixed = self.arg_list[self.Ntovary:]

        return zlist_fixed, lenslist_fixed, args_fixed

    def argsfixed_values(self):

        return np.array([args.values() for args in self.args_fixed])

    def argstovary_todictionary(self,values):

        args_list = []
        count = 0

        for n in range(0, int(self.Ntovary)):

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

        for n in range(0, int(self.Ntovary)):

            for key in self.routine.param_names[n]:

                if key not in self.routine.fixed_names[n]:
                    values.append(kwargs[n][key])

        return np.array(values)




