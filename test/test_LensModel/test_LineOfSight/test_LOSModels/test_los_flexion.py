__author__ = "nataliehogg"

import numpy as np
import numpy.testing as npt
import pytest
import unittest

from lenstronomy.LensModel.LineOfSight.LOSModels.los_flexion import LOSFlexion


class TestLOSFlexion(object):
    """Tests the LOS Flexion profile."""

    def setup_method(self):
        self.LOS = LOSFlexion()

    def test_set_static(self):

        p = self.LOS.set_static()

    def test_set_dynamic(self):

        d = self.LOS.set_dynamic()


if __name__ == "__main__":
    pytest.main("-k TestLOS")
