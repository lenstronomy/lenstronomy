
import pytest
import lenstronomy.LightModel.Profiles.torus as torus


class TestTorus(object):
    """
    class to test Shapelets
    """
    def setup(self):
        pass

    def test_function(self):
        """

        :return:
        """
        output = torus.function(x=1, y=1, amp=1., sigma=2, center_x=0, center_y=0)
        assert output == 0.079577471545947673


if __name__ == '__main__':
    pytest.main()
