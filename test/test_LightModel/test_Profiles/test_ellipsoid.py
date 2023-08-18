import pytest
import numpy as np
import lenstronomy.LightModel.Profiles.ellipsoid as torus
from lenstronomy.LightModel.Profiles.ellipsoid import Ellipsoid


class TestTorus(object):
    """
    class to test Shapelets
    """

    def setup_method(self):
        pass

    def test_function(self):
        """

        :return:
        """
        output = torus.function(x=1, y=1, amp=1.0, sigma=2, center_x=0, center_y=0)
        assert output == 0.079577471545947673


class TestEllipsoid(object):
    def setup_method(self):
        pass

    def test_function(self):
        """

        :return:
        """
        ellipsoid = Ellipsoid()
        output = ellipsoid.function(
            x=1, y=1, amp=1.0, radius=1, e1=0, e2=0, center_x=0, center_y=0
        )
        assert output == 0
        output = ellipsoid.function(
            x=0.99, y=0, amp=1.0, radius=1, e1=0, e2=0, center_x=0, center_y=0
        )
        assert output == 1.0 / np.pi


if __name__ == "__main__":
    pytest.main()
