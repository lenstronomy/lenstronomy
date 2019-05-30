import pytest
import numpy.testing as npt
from lenstronomy.LightModel.Profiles.interpolation import Interpol
from lenstronomy.LightModel.Profiles.gaussian import Gaussian
import lenstronomy.Util.util as util

class TestInterpol(object):
    """
    class to test Shapelets
    """
    def setup(self):
        pass

    def test_function(self):
        """

        :return:
        """
        x, y = util.make_grid(numPix=20, deltapix=1.)
        gauss = Gaussian()
        flux = gauss.function(x, y, amp=1., center_x=0., center_y=0., sigma_x=1., sigma_y=1.)
        image = util.array2image(flux)
        interp = Interpol()
        kwargs_interp = {'image': image, 'scale': 1., 'phi_G': 0., 'center_x': 0., 'center_y': 0.}
        output = interp.function(x, y, **kwargs_interp)
        npt.assert_almost_equal(output, flux, decimal=0)

        flux = gauss.function(x-1., y, amp=1., center_x=0., center_y=0., sigma_x=1., sigma_y=1.)
        kwargs_interp = {'image': image, 'scale': 1., 'phi_G': 0., 'center_x': 1., 'center_y': 0.}
        output = interp.function(x, y, **kwargs_interp)
        npt.assert_almost_equal(output, flux, decimal=0)

        flux = gauss.function(x - 1., y - 1., amp=1, center_x=0., center_y=0., sigma_x=1., sigma_y=1.)
        kwargs_interp = {'image': image, 'scale': 1., 'phi_G': 0., 'center_x': 1., 'center_y': 1.}
        output = interp.function(x, y, **kwargs_interp)
        npt.assert_almost_equal(output, flux, decimal=0)

        out = interp.function(x=1000, y=0, **kwargs_interp)
        assert out == 0


if __name__ == '__main__':
    pytest.main()
