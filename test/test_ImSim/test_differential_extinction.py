from lenstronomy.ImSim.differential_extinction import DifferentialExtinction
import pytest
import numpy.testing as npt


class TestDifferentialExtinction(object):

    def setup(self):
        pass

    def test_extinction(self):
        extinction = DifferentialExtinction(optical_depth_model=['GAUSSIAN'], tau0_index=0)
        kwargs_extinction = [{'amp': 1, 'sigma': 1, 'center_x': 0, 'center_y': 0}]
        kwargs_special = {'tau0_list': [2, 0]}
        ext = extinction.extinction(x=1, y=1, kwargs_special=kwargs_special, kwargs_extinction=kwargs_extinction)
        npt.assert_almost_equal(ext, 0.8894965388088921, decimal=8)

        ext = extinction.extinction(x=1, y=1, kwargs_special=kwargs_special, kwargs_extinction=None)
        npt.assert_almost_equal(ext, 1, decimal=8)

        ext = extinction.extinction(x=1, y=1, kwargs_special={}, kwargs_extinction=kwargs_extinction)
        npt.assert_almost_equal(ext, 0.9431312415612645, decimal=8)


if __name__ == '__main__':
    pytest.main()
