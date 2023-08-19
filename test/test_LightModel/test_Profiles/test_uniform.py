import pytest
from lenstronomy.LightModel.Profiles.uniform import Uniform


class TestShapelet(object):
    """Class to test Shapelets."""

    def setup_method(self):
        pass

    def test_function(self):
        """

        :return:
        """
        uniform = Uniform()
        output = uniform.function(x=1, y=1, amp=0.1)
        assert output == 0.1


if __name__ == "__main__":
    pytest.main()
