import pytest
import unittest

from lenstronomy.PointSource.Types.base_ps import PSBase
from lenstronomy.LensModel.lens_model import LensModel


class TestPSBase(object):

    def setup(self):
        self.base = PSBase(lens_model=LensModel(lens_model_list=[]), fixed_magnification=False, additional_image=False)
        PSBase(fixed_magnification=True, additional_image=True)

    def test_update_lens_model(self):
        self.base.update_lens_model(lens_model_class=None)
        assert self.base._solver is None

        base = PSBase()
        base.update_lens_model(lens_model_class=LensModel(lens_model_list=['SIS']))
        assert base._solver is not None
        PSBase(fixed_magnification=True, additional_image=True)


class TestUtil(object):

    def setup(self):
        pass

    def test_expand_to_array(self):
        from lenstronomy.PointSource.Types.base_ps import _expand_to_array
        array = 1
        num = 3
        array_out = _expand_to_array(array, num)
        assert len(array_out) == num

        array = [1]
        num = 3
        array_out = _expand_to_array(array, num)
        assert len(array_out) == num
        assert array_out[1] == 0

        array = [1, 1, 1]
        num = 3
        array_out = _expand_to_array(array, num)
        assert len(array_out) == num
        assert array_out[1] == 1


class TestRaise(unittest.TestCase):

    def test_raise(self):
        base = PSBase()
        with self.assertRaises(ValueError):
            base.image_position(kwargs_ps=None)
        with self.assertRaises(ValueError):
            base.source_position(kwargs_ps=None)
        with self.assertRaises(ValueError):
            base.image_amplitude(kwargs_ps=None)
        with self.assertRaises(ValueError):
            base.source_amplitude(kwargs_ps=None)


if __name__ == '__main__':
    pytest.main()
