from lenstronomy.SimulationAPI.observation_api import SingleBand
from lenstronomy.Data.imaging_data import ImageData
import lenstronomy.Util.util as util
import numpy as np

__all__ = ["DataAPI"]


class DataAPI(SingleBand):
    """
    This class is a wrapper of the general description of data in SingleBand() to translate those quantities into
    configurations in the core lenstronomy Data modules to simulate images according to those quantities.
    This class is meant to be an example of a wrapper. More possibilities in terms of PSF and data type
    options are available. Have a look in the specific modules if you are interested in.

    """

    def __init__(self, numpix, kwargs_pixel_grid=None, **kwargs_single_band):
        """

        :param numpix: number of pixels per axis in the simulation to be modelled
        :param kwargs_pixel_grid: if None, uses default pixel grid option
            if defined, must contain keyword arguments PixelGrid() class
        :param kwargs_single_band: keyword arguments used to create instance of SingleBand class
        """
        self.numpix = numpix
        if kwargs_pixel_grid is not None:
            required_keys = ["ra_at_xy_0", "dec_at_xy_0", "transform_pix2angle"]
            if not all(k in kwargs_pixel_grid for k in required_keys):
                raise ValueError(
                    "Missing 1 or more required" + "kwargs_pixel_grid parameters"
                )
        self._kwargs_pixel_grid = kwargs_pixel_grid
        SingleBand.__init__(self, **kwargs_single_band)

    @property
    def data_class(self):
        """
        creates a Data() instance of lenstronomy based on knowledge of the observation

        :return: instance of Data() class
        """
        data_class = ImageData(**self.kwargs_data)
        return data_class

    @property
    def kwargs_data(self):
        """

        :return: keyword arguments for ImageData class instance
        """
        # default pixel grid
        if self._kwargs_pixel_grid is None:
            (
                _,
                _,
                ra_at_xy_0,
                dec_at_xy_0,
                _,
                _,
                transform_pix2angle,
                _,
            ) = util.make_grid_with_coordtransform(
                numPix=self.numpix,
                deltapix=self.pixel_scale,
                subgrid_res=1,
                left_lower=False,
                inverse=False,
            )
        # user defined pixel grid
        else:
            ra_at_xy_0 = self._kwargs_pixel_grid["ra_at_xy_0"]
            dec_at_xy_0 = self._kwargs_pixel_grid["dec_at_xy_0"]
            transform_pix2angle = self._kwargs_pixel_grid["transform_pix2angle"]
        # CCD gain corrected exposure time to allow a direct Poisson estimates based on IID counts
        scaled_exposure_time = self.flux_iid(1)
        kwargs_data = {
            "image_data": np.zeros((self.numpix, self.numpix)),
            "ra_at_xy_0": ra_at_xy_0,
            "dec_at_xy_0": dec_at_xy_0,
            "transform_pix2angle": transform_pix2angle,
            "background_rms": self.background_noise,
            "exposure_time": scaled_exposure_time,
        }
        return kwargs_data
