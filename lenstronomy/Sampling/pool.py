import schwimmbad


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
    if processes == 1 or mpi:
        pool = schwimmbad.choose_pool(mpi=mpi, processes=1, **kwargs)
        is_master = pool.is_master()
    else:
        if 'use_dill' in kwargs:
            # schwimmbad MultiPool does not support dill so we remove this option from the kwargs
            _ = kwargs.pop('use_dill')
        pool = schwimmbad.choose_pool(mpi=False, processes=processes, **kwargs)
        # this MultiPool has no is_master() attribute like the SerialPool and MpiPool
        # all threads will then be 'master'.
        is_master = True
    return pool, is_master
