import numpy as np
import pytest

from lenstronomy.Sampling.Pool import parallelization_util as pu


class _FakeSamplerLikelihood(object):
    def logL(self, args):
        return float(np.sum(args))


class _FakeNestedLikelihood(object):
    def __call__(self, p):
        return float(np.sum(p))


class _FakeNonFiniteNestedLikelihood(object):
    def __call__(self, p):
        return np.nan


def setup_function():
    pu._SAMPLER_LIKELIHOOD_MODULE = None
    pu._NESTED_LIKELIHOOD_MODULE = None
    pu._NESTED_N_DIMS = None
    pu._NESTED_HAS_WARNED = False


def test_sampler_logl_worker_requires_initialization():
    with pytest.raises(RuntimeError):
        pu.sampler_logl_worker(np.array([1.0, 2.0]))


def test_sampler_logl_worker_evaluates_logl():
    pu.set_sampler_likelihood_module(_FakeSamplerLikelihood())
    result = pu.sampler_logl_worker(np.array([1.0, 2.0, 3.0]))
    assert result == 6.0


def test_nested_logl_worker_requires_initialization():
    with pytest.raises(RuntimeError):
        pu.nested_logl_worker(np.array([1.0, 2.0]))


def test_nested_logl_worker_finite_value_and_extra_args():
    pu.set_nested_likelihood_module(_FakeNestedLikelihood(), n_dims=2)
    result = pu.nested_logl_worker(np.array([1.5, 2.5, 99.0]), "unused")
    assert result == 4.0


def test_nested_logl_worker_non_finite_warns_once(capsys):
    pu.set_nested_likelihood_module(_FakeNonFiniteNestedLikelihood(), n_dims=2)

    result_1 = pu.nested_logl_worker(np.array([1.0, 2.0]))
    captured_1 = capsys.readouterr()

    result_2 = pu.nested_logl_worker(np.array([1.0, 2.0]))
    captured_2 = capsys.readouterr()

    assert result_1 == -1e15
    assert result_2 == -1e15
    assert "WARNING : logL is not finite" in captured_1.out
    assert captured_2.out == ""
