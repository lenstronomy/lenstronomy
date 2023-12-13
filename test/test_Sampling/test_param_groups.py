__author__ = "jhodonnell"


import numpy as np
import numpy.testing as npt
import unittest
import pytest

from lenstronomy.Sampling.param_group import ModelParamGroup, SingleParam, ArrayParam


class ExampleSingleParam(SingleParam):
    param_names = ["sp1", "sp2"]
    _kwargs_lower = {"sp1": 0, "sp2": 0}
    _kwargs_upper = {"sp1": 10, "sp2": 10}


class ExampleArrayParam(ArrayParam):
    param_names = {"ap1": 1, "ap2": 3}
    _kwargs_lower = {"ap1": [0], "ap2": [0] * 3}
    _kwargs_upper = {"ap1": [10], "ap2": [10] * 3}


class TestParamGroup(object):
    def setup_method(self):
        pass

    def test_single_param(self):
        sp = ExampleSingleParam(on=True)

        num, names = sp.num_params({})
        assert num == 2
        assert names == ["sp1", "sp2"]

        result = sp.set_params({"sp1": 2}, {"sp2": 3})
        assert result == [2]

        result = sp.set_params({"sp1": 2, "sp2": 3}, {})
        assert result == [2, 3]

        kwargs, i = sp.get_params(result, i=0, kwargs_fixed={})
        assert kwargs["sp1"] == 2
        assert kwargs["sp2"] == 3

    def test_array_param(self):
        ap = ExampleArrayParam(on=True)

        num, names = ap.num_params({})
        assert num == 4
        assert names == ["ap1"] + ["ap2"] * 3

        result = ap.set_params({"ap1": [2]}, {"ap2": [1, 1, 1]})
        assert result == [2]

        result = ap.set_params({"ap1": [2], "ap2": [1, 2, 3]}, {})
        assert result == [2, 1, 2, 3]

        kwargs, i = ap.get_params(result, i=0, kwargs_fixed={})
        assert kwargs["ap1"] == [2]
        assert kwargs["ap2"] == [1, 2, 3]

    def test_compose(self):
        sp = ExampleSingleParam(on=True)
        ap = ExampleArrayParam(on=True)

        num, names = ModelParamGroup.compose_num_params([sp, ap], kwargs_fixed={})
        assert num == 6
        assert names == sp.num_params({})[1] + ap.num_params({})[1]

        result = ModelParamGroup.compose_set_params(
            [sp, ap],
            {"sp1": 1, "sp2": 2, "ap1": [3], "ap2": [4, 5, 6]},
            kwargs_fixed={},
        )
        assert result == [1, 2, 3, 4, 5, 6]

        kwargs, i = ModelParamGroup.compose_get_params(
            [sp, ap], result, i=0, kwargs_fixed={}
        )
        assert kwargs["sp1"] == 1
        assert kwargs["ap2"] == [4, 5, 6]


if __name__ == "__main__":
    pytest.main()
