import types
import numpy as np

from lenstronomy.Sampling.Samplers.nautilus import Nautilus


class _FakeParam(object):
    @staticmethod
    def num_param():
        return 2, None

    @staticmethod
    def param_limits():
        return np.array([-1.0, -2.0]), np.array([1.0, 2.0])


class _FakeLikelihoodModule(object):
    def __init__(self):
        self.param = _FakeParam()

    @staticmethod
    def likelihood(a):
        return -float(np.sum(np.square(a)))


class _FakePool(object):
    @staticmethod
    def is_master():
        return True


class _FakePrior(object):
    def __init__(self):
        self.params = []

    def add_parameter(self, dist):
        self.params.append(dist)


class _FakeSampler(object):
    def __init__(self, prior, likelihood, pool, pass_dict=False, **kwargs):
        self.prior = prior
        self._likelihood = likelihood

    def add_bound(self):
        return None

    def fill_bound(self):
        return None

    @staticmethod
    def posterior(return_as_dict=False):
        points = np.zeros((2, 2))
        log_w = np.zeros(2)
        log_l = np.zeros(2)
        return points, log_w, log_l

    @staticmethod
    def evidence():
        return 0.0


def test_nautilus_sampling_uses_choose_pool(monkeypatch):
    def _fake_choose_pool(mpi, processes):
        assert mpi is False
        assert processes == 3
        return _FakePool()

    fake_nautilus_module = types.SimpleNamespace(Prior=_FakePrior, Sampler=_FakeSampler)

    monkeypatch.setattr(
        "lenstronomy.Sampling.Samplers.nautilus.choose_pool", _fake_choose_pool
    )
    monkeypatch.setitem(__import__("sys").modules, "nautilus", fake_nautilus_module)

    wrapper = Nautilus(_FakeLikelihoodModule())
    points, log_w, log_l, log_z = wrapper.nautilus_sampling(
        prior_type="uniform", mpi=False, thread_count=3, one_step=True
    )

    assert points.shape == (2, 2)
    assert len(log_w) == 2
    assert len(log_l) == 2
    assert log_z == 0.0
