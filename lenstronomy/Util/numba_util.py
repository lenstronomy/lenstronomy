from lenstronomy.Conf import config_loader
from os import environ

"""
From pyautolens:
Depending on if we're using a super computer, we want two different numba decorators:
If on laptop:
@numba.jit(nopython=True, cache=True, parallel=False)
If on super computer:
@numba.jit(nopython=True, cache=False, parallel=True)
"""

numba_conf = config_loader.numba_conf()
nopython = numba_conf["nopython"]
cache = numba_conf["cache"]
parallel = numba_conf["parallel"]
numba_enabled = numba_conf["enable"] and not environ.get("NUMBA_DISABLE_JIT", False)
fastmath = numba_conf["fastmath"]
error_model = numba_conf["error_model"]

if numba_enabled:
    try:
        import numba
        from numba import extending
    except ImportError:
        numba_enabled = False
        numba = None
        extending = None

__all__ = ["jit"]


def jit(
    nopython=nopython,
    cache=cache,
    parallel=parallel,
    fastmath=fastmath,
    error_model=error_model,
    inline="never",
):
    if numba_enabled:

        def wrapper(func):
            return numba.jit(
                func,
                nopython=nopython,
                cache=cache,
                parallel=parallel,
                fastmath=fastmath,
                error_model=error_model,
                inline=inline,
            )

    else:

        def wrapper(func):
            return func

    return wrapper
