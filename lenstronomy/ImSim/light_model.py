__author__ = 'sibirrer'

#this file contains a class which describes the surface brightness of the lens light

import numpy as np


class LensLightModel(object):

    def __init__(self, kwargs_options):
        lens_light_type = kwargs_options.get('lens_light_type', 'NONE')
        self.lightModel = LightModel(lens_light_type)

    def surface_brightness(self, x=0, y=0, **kwargs_lens_light):
        """
        :param x: coordinate in units of arcsec relative to the center of the image
        :type x: set or single 1d numpy array
        """
        return self.lightModel.surface_brightness(x, y, **kwargs_lens_light)


class SourceModel(object):

    def __init__(self, kwargs_options):
        source_type = kwargs_options.get('source_type', 'NONE')
        self.lightModel = LightModel(source_type)

    def surface_brightness(self, x=0, y=0, **kwargs_source):
        """
        :param x: coordinate in units of arcsec relative to the center of the image
        :type x: set or single 1d numpy array
        """
        return self.lightModel.surface_brightness(x, y, **kwargs_source)


class LightModel(object):
    """
    class to handle source and lens light models
    """
    def __init__(self, profile_type):
        self.valid = True
        if profile_type == 'GAUSSIAN':
            from astrofunc.LensingProfiles.gaussian import Gaussian
            self.func = Gaussian()
        elif profile_type == 'SERSIC':
            from astrofunc.LightProfiles.sersic import Sersic
            self.func = Sersic()
        elif profile_type == 'SERSIC_ELLIPSE':
            from astrofunc.LightProfiles.sersic import Sersic_elliptic
            self.func = Sersic_elliptic()
        elif profile_type == 'SHAPELETS':
            from astrofunc.LensingProfiles.shapelets import Shapelets
            self.func = Shapelets()
        elif profile_type == 'DOUBLE_SERSIC':
            from astrofunc.LightProfiles.sersic import DoubleSersic
            self.func = DoubleSersic()
        elif profile_type == 'CORE_SERSIC':
            from astrofunc.LightProfiles.sersic import CoreSersic
            self.func = CoreSersic()
        elif profile_type == 'DOUBLE_CORE_SERSIC':
            from astrofunc.LightProfiles.sersic import DoubleCoreSersic
            self.func = DoubleCoreSersic()
        elif profile_type == 'TRIPPLE_SERSIC':
            from astrofunc.LightProfiles.sersic import TrippleSersic
            self.func = TrippleSersic()
        elif profile_type == 'NONE':
            self.valid = False
        else:
            print('Warning! No lens light model of type', profile_type, ' found!')
            self.valid = False

    def surface_brightness(self, x=0, y=0, **kwargs):
        """
        :param x: coordinate in units of arcsec relative to the center of the image
        :type x: set or single 1d numpy array
        """
        if not self.valid:
            numPix = len(x)
            return np.zeros(numPix)
        else:
            return self.func.function(x, y, **kwargs)