__author__ = 'sibirrer'

#this file contains a class which describes the surface brightness of the source

class SourceModel(object):

    def __init__(self, kwargs_options):
        if kwargs_options['source_type'] == 'GAUSSIAN':
            from astrofunc.LensingProfiles.gaussian import Gaussian
            self.func = Gaussian()
        elif kwargs_options['source_type'] == 'SERSIC':
            from astrofunc.LightProfiles.sersic import Sersic
            self.func = Sersic()
        elif kwargs_options['source_type'] == 'SERSIC_ELLIPSE':
            from astrofunc.LightProfiles.sersic import Sersic_elliptic
            self.func = Sersic_elliptic()
        elif kwargs_options['source_type'] == 'SHAPELETS':
            from astrofunc.LensingProfiles.shapelets import Shapelets
            self.func = Shapelets()
        elif kwargs_options['source_type'] == 'NONE':
            pass
        else:
            raise ValueError('options do not include a valid source model!')

    def surface_brightness(self, x, y, **kwargs_source):
        """
        :param x: coordinate in units of arcsec relative to the center of the image
        :type x: set or single 1d numpy array
        """
        return self.func.function(x, y, **kwargs_source)
