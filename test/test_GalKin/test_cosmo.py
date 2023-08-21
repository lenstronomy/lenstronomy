from lenstronomy.GalKin.cosmo import Cosmo

import pytest
import unittest


class TestRaise(unittest.TestCase):

    def test_raise(self):
        with self.assertRaises(ValueError):
            Cosmo(d_d=-1, d_s=1, d_ds=1)


if __name__ == '__main__':
    pytest.main()
