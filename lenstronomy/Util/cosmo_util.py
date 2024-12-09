from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import LambdaCDM
from astropy.cosmology import FlatwCDM
from astropy.cosmology import wCDM
from astropy.cosmology import Flatw0waCDM
from astropy.cosmology import w0waCDM
from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()


@export
def get_astropy_cosmology(cosmology_model="FlatLambdaCDM", param_kwargs={}):
    """Return an instance of a astropy.cosmology class.

    :param cosmology_model: string, name of the cosmology model
    :type cosmology_model: str
    :param param_kwargs: keyword arguments of the cosmology class
    :type param_kwargs: dict
    :return: instance of a astropy.cosmology class
    """

    H0 = param_kwargs.get("H0", 70)
    Om0 = param_kwargs.get("Om0", 0.3)
    Ode0 = param_kwargs.get("Ode0", 0.7)
    w0 = param_kwargs.get("w0", -1)
    wa = param_kwargs.get("wa", 0)

    supported_models = [
        "FlatLambdaCDM",
        "LambdaCDM",
        "FlatwCDM",
        "wCDM",
        "Flatw0waCDM",
        "w0waCDM",
    ]

    if cosmology_model not in supported_models:
        raise ValueError(
            f"Cosmology model {cosmology_model} not supported! Choose from {supported_models}."
        )

    cosmo_classes = [FlatLambdaCDM, LambdaCDM, FlatwCDM, wCDM, Flatw0waCDM, w0waCDM]
    cosmo_kwargs = [
        {"H0": H0, "Om0": Om0},
        {"H0": H0, "Om0": Om0, "Ode0": Ode0},
        {"H0": H0, "Om0": Om0, "w0": w0},
        {"H0": H0, "Om0": Om0, "Ode0": Ode0, "w0": w0},
        {"H0": H0, "Om0": Om0, "w0": w0, "wa": wa},
        {"H0": H0, "Om0": Om0, "Ode0": Ode0, "w0": w0, "wa": wa},
    ]

    index = supported_models.index(cosmology_model)
    cosmo = cosmo_classes[index](**cosmo_kwargs[index])

    return cosmo
