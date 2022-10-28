from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
import unittest


class TestBaseProfile(object):

    def setup(self):
        pass

    def test_base_functions(self):
        base = LensProfileBase()
        base.set_static()
        base.set_dynamic()


class TestRaise(unittest.TestCase):

    def test_raise(self):
        base = LensProfileBase()
        with self.assertRaises(ValueError):
            base.function()
        with self.assertRaises(ValueError):
            base.derivatives()
        with self.assertRaises(ValueError):
            base.hessian()
        with self.assertRaises(ValueError):
            base.density_lens()
        with self.assertRaises(ValueError):
            base.mass_3d_lens()
        with self.assertRaises(ValueError):
            base.mass_2d_lens()
