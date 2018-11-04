import numpy as np
from lenstronomy.Util.param_util import phi_gamma_ellipticity,ellipticity2phi_gamma,phi_q2_ellipticity,\
    ellipticity2phi_q
from lenstronomy.Util.util import approx_theta_E

class FixedPowerLaw_Shear(object):

    def __init__(self,lens_model_list,kwargs_lens,xpos,ypos):

        assert lens_model_list[0] in ['SPEMD','SPEP']
        assert lens_model_list[1] == 'SHEAR'
        self._Ntovary = 2
        self._k_start = 2
        self._kwargs_lens = kwargs_lens

        self._theta_E_start = approx_theta_E(xpos, ypos)

        self._tovary_indicies = [0, 1]
        self.param_names = [['theta_E', 'center_x', 'center_y', 'e1', 'e2','gamma'], ['e1', 'e2']]
        self.fixed_names = [['gamma'], []]
        self.fixed_values = [{'gamma': kwargs_lens[0]['gamma']}, {}]
        self.params_to_vary = ['theta_E', 'center_x', 'center_y', 'e1', 'e2','shear_e1','shear_e2']

    def _new_ellip(self,start_e1,start_e2,delta_phi,delta_gamma):

        phi_start, gamma_start = ellipticity2phi_q(start_e1,start_e2)

        phi_min,phi_max = phi_start + delta_phi, phi_start-delta_phi
        gamma_min,gamma_max = max(0.001,gamma_start-delta_gamma),min(0.99,gamma_start+delta_gamma)

        e1_min, e2_min = phi_q2_ellipticity(phi_min, gamma_min)
        e1_max, e2_max = phi_q2_ellipticity(phi_max, gamma_max)

        return e1_min,e2_min,e1_max,e2_max

    def _new_shear(self, start_e1, start_e2, delta_phi, delta_gamma):

        phi_start, gamma_start = ellipticity2phi_gamma(start_e1, start_e2)

        phi_min, phi_max = phi_start + delta_phi, phi_start - delta_phi

        gamma_min, gamma_max = max(0.0001,gamma_start - delta_gamma), gamma_start + delta_gamma

        e1_min, e2_min = phi_gamma_ellipticity(phi_min, gamma_min)
        e1_max, e2_max = phi_gamma_ellipticity(phi_max, gamma_max)

        return e1_min, e2_min, e1_max, e2_max


    def get_param_ranges(self,reoptimize=False, scale = 1):

        if reoptimize:

            delta_phi,delta_ellip = scale * 30*np.pi*180**-1, scale * 0.15
            delta_shear_phi,delta_shear = scale * 30*np.pi*180**-1, scale * 0.02

            low_e1,low_e2, high_e1,high_e2  = self._new_ellip(self._kwargs_lens[0]['e1'], self._kwargs_lens[0]['e2'],
                                                              delta_phi, delta_ellip)

            low_shear_e1,low_shear_e2,high_shear_e1,high_shear_e2 = self._new_shear(self._kwargs_lens[1]['e1'],
                                                                                    self._kwargs_lens[1]['e2'],
                                                                                    delta_shear_phi, delta_shear)
            theta_E = scale * 0.005
            center = 0.005

            low_Rein = self._kwargs_lens[0]['theta_E'] - theta_E
            hi_Rein = self._kwargs_lens[0]['theta_E'] + theta_E

            low_centerx = self._kwargs_lens[0]['center_x'] - center
            hi_centerx = self._kwargs_lens[0]['center_x'] + center
            low_centery = self._kwargs_lens[0]['center_y'] - center
            hi_centery = self._kwargs_lens[0]['center_y'] + center

        else:

            low_e1 = -0.3
            low_e2 = low_e1
            high_e1 = 0.3
            high_e2 = high_e1

            low_shear_e1 = -0.08
            high_shear_e1 = 0.08
            low_shear_e2 = low_shear_e1
            high_shear_e2 = high_shear_e1

            low_Rein = self._theta_E_start - 0.1
            hi_Rein = self._theta_E_start + 0.1

            low_centerx = -0.015
            hi_centerx = 0.015
            low_centery = low_centerx
            hi_centery = hi_centerx

        sie_list_low = [low_Rein, low_centerx, low_centery, low_e1, low_e2]
        sie_list_high = [hi_Rein, hi_centerx, hi_centery, high_e1, high_e2]
        shear_list_low = [low_shear_e1, low_shear_e2]
        shear_list_high = [high_shear_e1, high_shear_e2]

        return sie_list_low+shear_list_low,sie_list_high+shear_list_high





