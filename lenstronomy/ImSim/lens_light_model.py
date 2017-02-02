__author__ = 'sibirrer'

#this file contains a class which describes the surface brightness of the lens light

import astrofunc.util as util
import numpy as np


class LensLightModel(object):

    def __init__(self, kwargs_options, data_kwargs):
        if 'lens_light_type' in kwargs_options:
            self.valid = True
            self.lens_light_type = kwargs_options['lens_light_type']
            if kwargs_options['lens_light_type'] == 'GAUSSIAN':
                from astrofunc.LensingProfiles.gaussian import Gaussian
                self.func = Gaussian()
            elif kwargs_options['lens_light_type'] == 'SERSIC':
                from astrofunc.LightProfiles.sersic import Sersic
                self.func = Sersic()
            elif kwargs_options['lens_light_type'] == 'SERSIC_ELLIPSE':
                from astrofunc.LightProfiles.sersic import Sersic_elliptic
                self.func = Sersic_elliptic()
            elif kwargs_options['lens_light_type'] == 'SHAPELETS':
                from astrofunc.LensingProfiles.shapelets import Shapelets
                self.func = Shapelets()
            elif kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC':
                from astrofunc.LightProfiles.sersic import DoubleSersic
                self.func = DoubleSersic()
            elif kwargs_options['lens_light_type'] == 'CORE_SERSIC':
                from astrofunc.LightProfiles.sersic import CoreSersic
                self.func = CoreSersic()
            elif kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC':
                from astrofunc.LightProfiles.sersic import DoubleCoreSersic
                self.func = DoubleCoreSersic()
            elif kwargs_options['lens_light_type'] == 'TRIPLE_SERSIC':
                from astrofunc.LightProfiles.sersic import TripleSersic
                self.func = TripleSersic()
            elif kwargs_options['lens_light_type'] == 'fixed':
                self.lens_light_model_fixed = data_kwargs['lens_light_model']
            elif kwargs_options['lens_light_type'] == 'NONE':
                self.valid = False
            else:
                print 'Warning! No lens light model of type', kwargs_options['lens_light_type'], ' found!'
                self.valid = False
        else:
            self.valid = False

    def surface_brightness(self, x=0, y=0, **kwargs_lens_light):
        """
        :param x: coordinate in units of arcsec relative to the center of the image
        :type x: set or single 1d numpy array
        """
        if not self.valid:
            numPix = len(x)
            return np.zeros(numPix)
        if self.lens_light_type == 'fixed':
            return util.image2array(self.lens_light_model_fixed)
        else:
            return self.func.function(x, y, **kwargs_lens_light)
