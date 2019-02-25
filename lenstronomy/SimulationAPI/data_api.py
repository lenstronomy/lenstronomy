from lenstronomy.SimulationAPI.observation_type import SingleBand
from lenstronomy.Data.imaging_data import Data
from lenstronomy.Data.psf import PSF
import lenstronomy.Util.util as util


class DataAPI(SingleBand):
    """
    This class is a wrapper of the general description of data in SingleBand() to translate those quantities into
    configurations in the core lenstronomy Data modules to simulate images according to those quantities.
    This class is meant to be an example of a wrapper. More possibilities in terms of PSF and data type
    options are available. Have a look in the specific modules if you are interested in.

    """
    def __init__(self, numpix, **kwargs_single_band):
        """

        :param numpix: number of pixels per axis in the simulation to be modelled
        :param kwargs_single_band: keyword arguments used to create instance of SingleBand class
        """
        self.numpix = numpix
        SingleBand.__init__(self, **kwargs_single_band)

    @property
    def data_class(self):
        """
        creates a Data() instance of lenstronomy based on knowledge of the observation

        :return: instance of Data() class
        """
        x_grid, y_grid, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix = util.make_grid_with_coordtransform(
            numPix=self.numpix, deltapix=self.pixel_scale, subgrid_res=1, left_lower=False, inverse=False)
        kwargs_data = {'numPix': self.numpix, 'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
                       'transform_pix2angle': Mpix2coord,
                       'background_rms': self.background_noise,
                       'exp_time': self.scaled_exposure_time}
        data_class = Data(kwargs_data)
        return data_class

    @property
    def psf_class(self):
        """
        creates instance of PSF() class based on knowledge of the observations
        For the full possibility of how to create such an instance, see the PSF() class documentation

        :return: instance of PSF() class
        """
        if self._psf_type == 'GAUSSIAN':
            psf_type = "GAUSSIAN"
            fwhm = self._seeing
            kwargs_psf = {'psf_type': psf_type, 'fwhm': fwhm}
        elif self._psf_type == 'PIXEL':
            if self._psf_model is not None:
                kwargs_psf = {'psf_type': "PIXEL", 'kernel_point_source': self._psf_model}
            else:
                raise ValueError("You need to create the class instance with a psf_model!")
        else:
            raise ValueError("psf_type %s not supported!" % self._psf_type)
        psf_class = PSF(kwargs_psf)
        return psf_class
