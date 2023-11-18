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


if __name__ == "__main__":
    pytest.main()
