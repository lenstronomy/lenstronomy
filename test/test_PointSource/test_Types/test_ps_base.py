import pytest


class TestUtil(object):

    def setup(self):
        pass

    def test_expand_t0_array(self):
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


if __name__ == '__main__':
    pytest.main()
