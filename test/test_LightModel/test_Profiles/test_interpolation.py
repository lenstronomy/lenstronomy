import pytest
import numpy.testing as npt
from lenstronomy.LightModel.Profiles.interpolation import Interpol
from lenstronomy.LightModel.Profiles.gaussian import Gaussian
import lenstronomy.Util.util as util
import numpy as np


class TestInterpol(object):
    """Class to test Shapelets."""
    def setup_method(self):
        pass

    def test_function(self):
        """

        :return:
        """
        for len_x, len_y in [(20, 20), (14, 20)]:
            x, y = util.make_grid(numPix=(len_x, len_y), deltapix=1.)
            gauss = Gaussian()
            flux = gauss.function(x, y, amp=1., center_x=0., center_y=0., sigma=1.)
            image = util.array2image(flux, nx=len_y, ny=len_x)

            interp = Interpol()
            kwargs_interp = {'image': image, 'scale': 1., 'phi_G': 0., 'center_x': 0., 'center_y': 0.}
            output = interp.function(x, y, **kwargs_interp)

            npt.assert_equal(output, flux)

            flux = gauss.function(x-1., y, amp=1., center_x=0., center_y=0., sigma=1.)
            kwargs_interp = {'image': image, 'scale': 1., 'phi_G': 0., 'center_x': 1., 'center_y': 0.}
            output = interp.function(x, y, **kwargs_interp)
            npt.assert_almost_equal(output, flux, decimal=0)

            flux = gauss.function(x - 1., y - 1., amp=1, center_x=0., center_y=0., sigma=1.)
            kwargs_interp = {'image': image, 'scale': 1., 'phi_G': 0., 'center_x': 1., 'center_y': 1.}
            output = interp.function(x, y, **kwargs_interp)
            npt.assert_almost_equal(output, flux, decimal=0)

            out = interp.function(x=1000, y=0, **kwargs_interp)
            assert out == 0

        # test change of center without re-doing interpolation
        out = interp.function(x=0, y=0, image=image, scale=1., phi_G=0, center_x=0, center_y=0)
        out_shift = interp.function(x=1, y=0, image=image, scale=1., phi_G=0, center_x=1, center_y=0)
        assert out_shift == out

        # function must give a single value when evaluated at a single point
        assert isinstance(out, float)

        # test change of scale without re-doing interpolation
        out = interp.function(x=1., y=0, image=image, scale=1., phi_G=0, center_x=0, center_y=0)
        out_scaled = interp.function(x=2., y=0, image=image, scale=2, phi_G=0, center_x=0, center_y=0)
        assert out_scaled == out

    def test_flux_normalization(self):
        interp = Interpol()
        delta_pix = 0.1
        len_x, len_y = 21, 21
        x, y = util.make_grid(numPix=(len_x, len_y), deltapix=delta_pix)
        gauss = Gaussian()
        flux = gauss.function(x, y, amp=1., center_x=0., center_y=0., sigma=0.3)
        image = util.array2image(flux, nx=len_y, ny=len_x)
        flux_total = np.sum(image)

        kwargs_interp = {'image': image, 'scale': delta_pix, 'phi_G': 0., 'center_x': 0., 'center_y': 0.}
        image_interp = interp.function(x, y, **kwargs_interp)
        flux_interp = np.sum(image_interp)
        npt.assert_almost_equal(flux_interp, flux_total, decimal=3)

        # test for scale !=1
        # demands same surface brightness values. We rescale the pixel grid by the same amount as the image
        scale = 0.5
        x, y = util.make_grid(numPix=(len_x, len_y), deltapix=delta_pix * scale)
        kwargs_interp = {'image': image, 'scale': delta_pix * scale, 'phi_G': 0., 'center_x': 0., 'center_y': 0.}
        output = interp.function(x, y, **kwargs_interp)

        npt.assert_almost_equal(output / image_interp, 1, decimal=5)

    def test_delete_cache(self):
        x, y = util.make_grid(numPix=20, deltapix=1.)
        gauss = Gaussian()
        flux = gauss.function(x, y, amp=1., center_x=0., center_y=0., sigma=1.)
        image = util.array2image(flux)
        interp = Interpol()
        kwargs_interp = {'image': image, 'scale': 1., 'phi_G': 0., 'center_x': 0., 'center_y': 0.}
        output = interp.function(x, y, **kwargs_interp)
        assert hasattr(interp, '_image_interp')
        interp.delete_cache()
        assert not hasattr(interp, '_image_interp')


if __name__ == '__main__':
    pytest.main()
