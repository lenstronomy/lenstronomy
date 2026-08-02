__author__ = "ajshajib"

import numpy as np

_SAMPLER_LIKELIHOOD_MODULE = None
_NESTED_LIKELIHOOD_MODULE = None
_NESTED_N_DIMS = None
_NESTED_HAS_WARNED = False


def set_sampler_likelihood_module(likelihood_module):
    """Set the likelihood module used by worker log-likelihood evaluations.

    :param likelihood_module: a likelihood module like in likelihood.py (should be
        callable)
    :return: None
    """
    global _SAMPLER_LIKELIHOOD_MODULE
    _SAMPLER_LIKELIHOOD_MODULE = likelihood_module


def sampler_logl_worker(args):
    """Evaluate the log-likelihood from worker-local sampler state.

    :param args: parameter vector for which to evaluate the log-likelihood
    :return: log-likelihood value
    """
    if _SAMPLER_LIKELIHOOD_MODULE is None:
        raise RuntimeError(
            "Worker likelihood module is not initialized. "
            "Call set_sampler_likelihood_module before evaluating logL."
        )
    return _SAMPLER_LIKELIHOOD_MODULE.logL(args)


def set_nested_likelihood_module(likelihood_module, n_dims):
    """Set the nested-sampler likelihood and dimensionality on each worker.

    :param likelihood_module: a likelihood module like in likelihood.py (should be
        callable)
    :param n_dims: number of dimensions in the parameter space
    :return: None
    """
    global _NESTED_LIKELIHOOD_MODULE, _NESTED_N_DIMS, _NESTED_HAS_WARNED
    _NESTED_LIKELIHOOD_MODULE = likelihood_module
    _NESTED_N_DIMS = n_dims
    _NESTED_HAS_WARNED = False


def nested_logl_worker(p, *args):
    """Worker log-likelihood entrypoint used by MPI nested samplers.

    :param p: parameter vector for which to evaluate the log-likelihood
    :param args: additional arguments (ignored)
    :return: log-likelihood value
    """
    if _NESTED_LIKELIHOOD_MODULE is None or _NESTED_N_DIMS is None:
        raise RuntimeError(
            "Worker nested likelihood is not initialized. "
            "Call set_nested_likelihood_module before evaluating logL."
        )

    p = np.array([p[i] for i in range(_NESTED_N_DIMS)])
    log_l = _NESTED_LIKELIHOOD_MODULE(p)

    global _NESTED_HAS_WARNED
    if not np.isfinite(log_l):
        if not _NESTED_HAS_WARNED:
            print("WARNING : logL is not finite : return very low value instead")
        log_l = -1e15
        _NESTED_HAS_WARNED = True
    return float(log_l)
