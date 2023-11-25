import unittest
import pytest
from lenstronomy.LightModel.Profiles.profile_base import LightProfileBase


class TestRaise(unittest.TestCase):
    def test_raise(self):
        lighModel = LightProfileBase()
        with self.assertRaises(ValueError):
            lighModel.function()
        with self.assertRaises(ValueError):
            lighModel.light_3d()


if __name__ == "__main__":
    pytest.main()
