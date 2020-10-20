# coding: utf-8
"""
this file is taken from schwimmbad (https://github.com/adrn/schwimmbad) and an explicit fork by Aymeric Galan
to replace the multiprocessing with the multiprocess dependence as for multi-threading, multiprocessing is
not supporting dill (only pickle) which is required.

Tests show that the MPI mode works with Python 3.7.2 but not with Python 3.7.0 on a specific system due to mpi4py
dependencies and configurations.


Contributions by:
- Peter K. G. Williams
- Júlio Hoffimann Mendes
- Dan Foreman-Mackey
- Aymeric Galan
- Simon Birrer

Implementations of four different types of processing pools:

    - MPIPool: An MPI pool.
    - MultiPool: A multiprocessing for local parallelization.
    - SerialPool: A serial pool, which uses the built-in `map` function

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

__all__ = ['choose_pool']


def choose_pool(mpi=False, processes=1, **kwargs):
    """
    Extends the capabilities of the schwimmbad.choose_pool method.

    It handles the `use_dill` parameters in kwargs, that would otherwise raise an error when processes > 1.
    Any thread in the returned multiprocessing pool (e.g. processes > 1) also default

    The requirement of schwimmbad relies on the master branch (as specified in requirements.txt).
    The 'use_dill' functionality can raise if not following the requirement specified.

    Choose between the different pools given options from, e.g., argparse.

    Parameters
    ----------
    mpi : bool, optional
        Use the MPI processing pool, :class:`~schwimmbad.mpi.MPIPool`. By
        default, ``False``, will use the :class:`~schwimmbad.serial.SerialPool`.
    processes : int, optional
        Use the multiprocessing pool,
        :class:`~schwimmbad.multiprocessing.MultiPool`, with this number of
        processes. By default, ``processes=1``, will use them:class:`~schwimmbad.serial.SerialPool`.

    Any additional kwargs are passed in to the pool class initializer selected by the arguments.

    """

    if mpi:
        if not MPIPool.enabled():
            raise SystemError("Tried to run with MPI but MPIPool not enabled.")
        try:
            pool = MPIPool(**kwargs)
        except:
            raise ImportError('MPIPool of schwimmbad can not be generated. lenstronomy uses a specific branch of '
                              'schwimmbad specified in the requirements.txt. Make sure you are using the correct '
                              'version of schwimmbad. In particular the "use_dill" argument is not supported in the '
                              'pypi version 0.3.0.')
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
