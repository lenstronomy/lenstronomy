from copy import deepcopy
from lenstronomy.Util.param_util import shear_cartesian2polar, shear_polar2cartesian
from lenstronomy.Util.param_util import ellipticity2phi_q


class PowerLawFreeShear(object):

    def __init__(self, kwargs_lens_init):

        self.kwargs_lens = kwargs_lens_init

    def update_kwargs_lens(self, kwargs_new):

        idx1 = 0
        self.kwargs_lens[idx1] = kwargs_new[0]
        idx2 = 1
        self.kwargs_lens[idx2] = kwargs_new[1]
        return self.kwargs_lens

    def args_to_kwargs(self, args):

        gamma = self.kwargs_lens[0]['gamma']
        kwargs_epl = {'theta_E': args[0], 'center_x': args[1], 'center_y': args[2],
                      'e1': args[3], 'e2': args[4], 'gamma': gamma}
        kwargs_shear = {'gamma1': args[5], 'gamma2': args[6]}

        kwargs_lens = self.update_kwargs_lens([kwargs_epl, kwargs_shear])

        return kwargs_lens

class PowerLawFixShear(object):

    def __init__(self, kwargs_lens_init, shear_strength):

        self.kwargs_lens = kwargs_lens_init
        self._inds_to_vary = [0, 1]
        self.shear_strength = shear_strength

    def update_kwargs_lens(self, kwargs_new):
        idx1 = self._inds_to_vary[0]
        self.kwargs_lens[idx1] = kwargs_new[0]
        idx2 = self._inds_to_vary[1]
        self.kwargs_lens[idx2] = kwargs_new[1]
        return self.kwargs_lens

    def args_to_kwargs(self, args):

        gamma = self.kwargs_lens[self._inds_to_vary[0]]['gamma']
        kwargs_epl = {'theta_E': args[0], 'center_x': args[1], 'center_y': args[2],
                      'e1': args[3], 'e2': args[4], 'gamma': gamma}

        phi, _ = shear_cartesian2polar(args[5], args[6])
        gamma1, gamma2 = shear_polar2cartesian(phi, self.shear_strength)
        kwargs_shear = {'gamma1': gamma1, 'gamma2': gamma2}

        kwargs_lens = self.update_kwargs_lens([kwargs_epl, kwargs_shear])

        return kwargs_lens

class PowerLawBoxyDiskyFreeShear(PowerLawFreeShear):


    def args_to_kwargs(self, args):

        gamma = self.kwargs_lens[0]['gamma']

        kwargs_epl = {'theta_E': args[0], 'center_x': args[1], 'center_y': args[2],
                      'e1': args[3], 'e2': args[4], 'gamma': gamma}

        kwargs_shear = {'gamma1': args[5], 'gamma2': args[6]}

        phi, _ = ellipticity2phi_q(args[3], args[4])

        phi_m = phi
        
    def update_kwargs_lens(self, kwargs_new):

        idx1 = self._inds_to_vary[0]
        self.kwargs_lens[idx1] = kwargs_new[0]
        idx2 = self._inds_to_vary[1]
        self.kwargs_lens[idx2] = kwargs_new[1]
        return self.kwargs_lens





