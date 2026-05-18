__author__ = "nataliehogg"

import numpy as np
import numpy.testing as npt
import pytest
import unittest

from lenstronomy.LensModel.LineOfSight.LOSModels.los_flexion_minimal import (
    LOSFlexionMinimal,
)


class TestLOSFlexionMinimal(object):
    """Tests the LOSFlexionMinimal profile; inherits from LOS so we can repeat those
    tests here.

    This is functionally redundant but boosts coverage.
    """

    def setup_method(self):
        self.LOSMinimal = LOSFlexionMinimal()


if __name__ == "__main__":
    pytest.main("-k TestLOSMinimal")
