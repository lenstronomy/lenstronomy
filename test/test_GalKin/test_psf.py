from lenstronomy.GalKin.psf import PSF
import numpy as np
import unittest


class TestPSF(object):

    def setup(self):
        pass

    def test_displace_psf(self):
        psf = PSF(psf_type='GAUSSIAN', fwhm=1)
        np.random.seed(41)
        x, y = psf.displace_psf(0, 0)
        assert x != 0
        assert y != 0

        psf = PSF(psf_type='MOFFAT', fwhm=1, moffat_beta=2.6)
        np.random.seed(41)
        x, y = psf.displace_psf(0, 0)
        assert x != 0
        assert y != 0


class TestRaise(unittest.TestCase):

    def test_raise(self):
        with self.assertRaises(ValueError):
            psf = PSF(psf_type='BRRR', fwhm=1, moffat_beta=2.6)
            #psf.displace_psf(0, 0)
