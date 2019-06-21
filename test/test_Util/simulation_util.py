# import main simulation class of lenstronomy
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
import pytest


class TestSimulation(object):
    def setup(self):
        pass

    def test_data_configure_simple(self):

        # data specifics
        sigma_bkg = 1.  # background noise per pixel
        exp_time = 10  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 100  # cutout pixel size
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg)
        data_class = ImageData(**kwargs_data)
        assert data_class.pixel_width == deltaPix


if __name__ == '__main__':
    pytest.main()
