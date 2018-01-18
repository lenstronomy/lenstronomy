
import pytest
from lenstronomy.LightModel.Profiles.moffat import Moffat


class TestMoffat(object):
    """
    class to test the Moffat profile
    """
    def setup(self):
        pass

    def test_function(self):
        """

        :return:
        """
        profile = Moffat()
        output = profile.function(x=1., y=1., amp=1., alpha=2., beta=1., center_x=0, center_y=0)
        assert output == 0.6666666666666666


if __name__ == '__main__':
    pytest.main()
