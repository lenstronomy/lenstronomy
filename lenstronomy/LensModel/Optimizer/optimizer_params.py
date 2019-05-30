from lenstronomy.LensModel.Optimizer.fixed_routines import *
import numpy as np


class Params(object):

    known_routines = ['fixed_powerlaw_shear', 'variable_powerlaw_shear', 'fixedshearpowerlaw']

    def __init__(self, zlist=None, lens_list=None, arg_list=None,
                 optimizer_routine=str, xpos = None, ypos = None,
                 constrain_params = None):

        if optimizer_routine not in self.known_routines:
            raise Exception('routine '+str(optimizer_routine)+' not recognized.')

        self.routine_name = optimizer_routine

        if optimizer_routine == 'fixed_powerlaw_shear':
            routine = FixedPowerLaw_Shear(lens_list,arg_list,xpos,ypos,constrain_params)
        elif optimizer_routine == 'variable_powerlaw_shear':
            routine = VariablePowerLaw_Shear(lens_list, arg_list, xpos, ypos,constrain_params)
        elif optimizer_routine == 'fixedshearpowerlaw':
            routine = FixedShearPowerLaw(lens_list, arg_list, xpos, ypos, constrain_params)

        self._zlist = zlist
        self._lens_list = lens_list
        self._arg_list = arg_list

        self._k_start = routine._k_start
        self._Ntovary = routine._Ntovary
        self.tovary_indicies = routine._tovary_indicies
        self.routine = routine

        self.zlist_tovary,self.lenslist_tovary,self.args_tovary = self.to_vary()
        self.zlist_fixed, self.lenslist_fixed, self.args_fixed = self.fixed()

    def to_vary_limits(self,re_optimize, scale = 1):

        lower_limit,upper_limit = self.routine.get_param_ranges(re_optimize, scale)

        return lower_limit,upper_limit

    def to_vary(self):
        zlist_tovary = self._zlist[0:self._Ntovary]
        lenslist_tovary = self._lens_list[0:self._Ntovary]
        args_tovary = self._arg_list[0:self._Ntovary]

        return zlist_tovary, lenslist_tovary, args_tovary

    def fixed(self):
        zlist_fixed = self._zlist[self._Ntovary:]
        lenslist_fixed = self._lens_list[self._Ntovary:]
        args_fixed = self._arg_list[self._Ntovary:]

        return zlist_fixed, lenslist_fixed, args_fixed

    def argstovary_todictionary(self,values):

        args_list = []
        count = 0

        for n in range(0, int(self._Ntovary)):

            args = {}

            if hasattr(self.routine, '_fixshear') and n == 1:
                # here values is [theta_E, x, y, e1, e2, shear_theta]
                # so convert into shear e1 e2
                e1, e2 = phi_gamma_ellipticity(values[count],
                                               self.routine.fixed_values[n]['shear_magnitude'])
                args.update({'e1': e1, 'e2': e2})

            else:
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

                if key == 'shear_magnitude' or key == 'shear_PA':

                    phi, _ = ellipticity2phi_gamma(kwargs[n]['e1'], kwargs[n]['e2'])
                    values.append(phi)

                else:
                    if key not in self.routine.fixed_names[n]:
                        values.append(kwargs[n][key])

        return np.array(values)




