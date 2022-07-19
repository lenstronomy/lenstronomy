import numpy as np
import Util.param_util as param_util

__all__ = ['KinLikelihood']


class KinLikelihood(object):
    """
    Class to compute hte likelihood associated to binned 2D kinematic maps
    """
    def __init__(self, kinematic_data_2D_class, lens_model_class, lens_light_model_class,kwargs_data,idx_lens=0,
                 idx_lens_light=0):
        """
        :param kinematic_data_2D_class: KinData class instance
        :param lens_model_class: LensModel class instance
        :param lens_light_model_class: LightModel class instance
        """
        self.lens_model_class = lens_model_class
        self._idx_lens = idx_lens
        self._idx_lens_light = idx_lens_light
        self.kin_input = kinematic_data_2D_class.KinBin2kwargs()
        self.image_input = kwargs_data2image_input(kwargs_data)

    def logL(self, kwargs_lens, kwargs_lens_light, kwargs_special):
        """
        Calculates Log likelihood from 2D kinematic likelihood
        """
        orientation_ellipse,q = param_util.ellipticity2phi_q(kwargs_lens[self._idx_lens]['e1'],
                                                             kwrags_lens[self._idx_lens]['e2'])
        self.image_input['ellipse_PA']=orientation_ellipse

        return 0
    def kwargs_data2image_input(self,kwargs_data):
        """
        Creates the kwargs of the image needed for 2D kinematic likelihood
        """
        deltaPix = np.sqrt(np.abs(np.linalg.det(kwargs_data['transform_pix2angle'])))
        kwargs = {'image':kwargs_data['imaging_data'], 'deltaPix':deltaPix, 'transform_pix2angle':deltaPix,
                  'ra_at_xy0':kwargs_data['ra_at_xy_0'],'dec_at_xy0':kwargs_data['dec_at_xy_0']}
        return kwargs