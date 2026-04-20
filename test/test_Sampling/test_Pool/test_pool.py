import pytest
from lenstronomy.Sampling.Pool.pool import choose_pool
from lenstronomy.Sampling.Pool.multiprocessing import MultiPool


class TestPool(object):
    def setup_method(self):
        pass

    def test_choose_pool(self):
        import schwimmbad

        pool = choose_pool(mpi=False, processes=1, use_dill=True)
        assert pool.is_master() is True
        assert isinstance(pool, schwimmbad.serial.SerialPool)

        pool = choose_pool(mpi=False, processes=2, use_dill=True)
        assert pool.is_master() is True
        assert isinstance(pool, MultiPool)

        # NOTE: MPI cannot be tested here (needs to be launched with mpirun)
        # pool = choose_pool(mpi=True, processes=1, use_dill=True)
        # assert pool.is_master() is True
        # assert isinstance(pool, schwimmbad.mpi.MPIPool)

    def test_choose_pool_mpi_init_error(self, monkeypatch):
        import schwimmbad

        class _FailingMPIPool(object):
            @staticmethod
            def enabled():
                return True

            def __init__(self, **kwargs):
                raise RuntimeError("mpi init failed")

        monkeypatch.setattr(schwimmbad, "MPIPool", _FailingMPIPool)

        with pytest.raises(SystemError, match="failed to initialize MPIPool"):
            choose_pool(mpi=True, processes=1)


if __name__ == "__main__":
    pytest.main()
