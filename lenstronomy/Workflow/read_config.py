__author__ = 'sibirrer'

import Config.config as Config
from lenstronomy.Cosmo.cosmo_properties import CosmoProp
from lenstronomy.strong_lens_data.strong_lens_system_factory import StrongLensSystemFactory
from lenstronomy.ImSim.make_image import MakeImage
from lenstronomy.Cosmo.unit_manager import UnitManager
from lenstronomy.Workflow.parameters import Param


class ConfigReader(object):
    """
    reads in a config file and deals with its options
    """
    def __init__(self):
        self.config = Config
        self.kwargs_options = {'system_name': Config.system_name, 'data_file': Config.data_file
            , 'cosmo_file': Config.cosmo_file, 'lens_type': Config.lens_type, 'source_type': Config.source_type
            , 'subgrid_res': Config.subgrid_res, 'numPix': Config.numPix, 'psf_type': Config.psf_type, 'x2_simple': Config.x2_simple}


class ConfigSetUp(object):
    """
    class which reads in kwargs_options and initialises the classes accordingly (but only if needed)
    """
    def __init__(self,kwargs_options):
        """
        load database
        load image to be matched from database
        initialize cosmology
        initialize LensSystemSim
        """
        self.kwargs_options = kwargs_options
        self.strongLensSystemFactory = StrongLensSystemFactory()
        attrname = 'data'
        self.system = self.create_from_central_add_image(kwargs_options['system_name'], attrname, fits_filename= kwargs_options['data_file'],cutout_filename=None, cutout_scale=Config.numPix)


    def create_from_central_add_image(self, system_name, attrname, fits_filename,cutout_filename=None, cutout_scale=None):
        """

        :param system_name: name of the lens system
        :param attrname: name of the data object to call
        :return: class of StrongLensSystemData with attached image in the right cutout
        """
        system = self.strongLensSystemFactory.find_from_central(system_name)
        ra = system.ra
        dec = system.dec
        print(ra,dec,'ra, dec in create_from_central_add_image in read_config')
        system.add_image_data(fits_filename,attrname,ra_cutout_cent=ra,dec_cutout_cent=dec,
                       load_initial=False, data_manager = None, cutout_filename = cutout_filename,cutout_scale = cutout_scale)
        return system


    def cosmo(self):
        if not hasattr(self, 'cosmoProp'):
            cosmo_file = Config.cosmo_file
            z_L = getattr(self.system,'z_lens')
            z_S = getattr(self.system,'z_source')
            self.cosmoProp = CosmoProp(z_L,z_S,cosmo_file)
        return self.cosmoProp

    def unit(self):
        if not hasattr(self, 'unitManager'):
            StrongLensImageData=getattr(self.system,'data')
            self.unitManager = UnitManager(self.cosmo(),StrongLensImageData)
        return self.unitManager

    def param(self):
        if not hasattr(self, 'param_class'):
            self.param_class = Param()
        return  self.param_class



    def make_image(self):
        if not hasattr(self, 'makeImage'):
            self.makeImage = MakeImage(self.lens(), self.source())
        return self.makeImage


    def mcmc(self):
        if not hasattr(self, 'mcmc_class'):
            data = self.system.get_cutout_image('data')
            pixScale1, pixScale2 = self.system.get_pixel_scale('data')
            self.mcmc_class = MCMC(self.make_image(),LensSystemSim(pixScale1),self.param(),data)
        return self.mcmc_class


class LensSystemSim(object):
    """
    class for parameters of the lens system which do not vary
    """
    def __init__(self,deltaPix):
        self.deltaPix = deltaPix