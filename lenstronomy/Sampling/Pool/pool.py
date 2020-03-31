# coding: utf-8
"""
Contributions by:
- Peter K. G. Williams
- Júlio Hoffimann Mendes
- Dan Foreman-Mackey

Implementations of four different types of processing pools:

    - MPIPool: An MPI pool.
    - MultiPool: A multiprocessing for local parallelization.
    - SerialPool: A serial pool, which uses the built-in ``map`` function

"""

__version__ = "0.3.0"
__author__ = "Adrian Price-Whelan <adrianmpw@gmail.com>"

# Standard library
import sys
import logging
log = logging.getLogger(__name__)
_VERBOSE = 5

#from schwimmbad.multiprocessing import MultiPool
from lenstronomy.Sampling.Pool.multiprocessing import MultiPool
from schwimmbad.serial import SerialPool
from schwimmbad.mpi import MPIPool
#from schwimmbad.jl import JoblibPool


def choose_pool(mpi=False, processes=1, **kwargs):
    """
    Extends the capabilities of the schwimmbad.choose_pool method.

    It handles the `use_dill` parameters in kwargs, that would otherwise raise an error when processes > 1.
    Any thread in the returned multiprocessing pool (e.g. processes > 1) also default


    Docstring from schwimmbad:

    mpi : bool, optional
        Use the MPI processing pool, :class:`~schwimmbad.mpi.MPIPool`. By
        default, ``False``, will use the :class:`~schwimmbad.serial.SerialPool`.
    processes : int, optional
        Use the multiprocessing pool,
        :class:`~schwimmbad.multiprocessing.MultiPool`, with this number of
        processes. By default, ``processes=1``, will use the
        :class:`~schwimmbad.serial.SerialPool`.
    **kwargs
            Any additional kwargs are passed in to the pool class initializer selected by the arguments.
    """
    pool = choose_pool_schwimmbad(mpi=mpi, processes=processes, **kwargs)
    is_master = pool.is_master()
    return pool, is_master


def choose_pool_schwimmbad(mpi=False, processes=1, **kwargs):
    """
    Choose between the different pools given options from, e.g., argparse.

    Parameters
    ----------
    mpi : bool, optional
        Use the MPI processing pool, :class:`~schwimmbad.mpi.MPIPool`. By
        default, ``False``, will use the :class:`~schwimmbad.serial.SerialPool`.
    processes : int, optional
        Use the multiprocessing pool,
        :class:`~schwimmbad.multiprocessing.MultiPool`, with this number of
        processes. By default, ``processes=1``, will use the
        :class:`~schwimmbad.serial.SerialPool`.
    **kwargs
        Any additional kwargs are passed in to the pool class initializer
        selected by the arguments.
    """

    if mpi:
        if not MPIPool.enabled():
            raise SystemError("Tried to run with MPI but MPIPool not enabled.")

        pool = MPIPool(**kwargs)
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        log.info("Running with MPI on {0} cores".format(pool.size))
        return pool

    elif processes != 1 and MultiPool.enabled():
        if 'use_dill' in kwargs:
            # schwimmbad MultiPool does not support dill so we remove this option from the kwargs
            _ = kwargs.pop('use_dill')
        log.info("Running with MultiPool on {0} cores".format(processes))
        return MultiPool(processes=processes, **kwargs)

    else:
        log.info("Running with SerialPool")
        return SerialPool(**kwargs)
