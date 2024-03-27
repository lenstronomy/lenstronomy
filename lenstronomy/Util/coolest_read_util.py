import numpy as np
import coolest
from lenstronomy.Util.param_util import phi_q2_ellipticity, shear_polar2cartesian


def shapelet_amp_coolest_to_lenstronomy(value):
    """Transforms shapelets coefficients from COOLEST conventions (x to the right) to
    lenstronomy conventions (x following ra, to the left)

    :param value: amplitude of the shapelet (float or np.array) in COOLEST conventions
    :return: amplitude of the shapelet (float or np.array) in lenstronomy conventions
    """
    if value is None:
        return None
    else:
        n = 0  # this is the shapelet order
        k = 0  # this is the index of a coefficient for a given order
        new_value = []
        for coeff in value:
            if n % 2 == 0:
                if k % 2 == 1:
                    coeff = -coeff
            elif n % 2 == 1:
                if k % 2 == 0:
                    coeff = -coeff
            new_value.append(coeff)
            k += 1
            if k == n + 1:
                n += 1
                k = 0
        return new_value


def degree_coolest_to_radian_lenstronomy(value):
    """
    Transform an angle in degree in COOLEST conventions (from y to negative x - aka East of North)
    into an angle in radian in lenstronomy conventions (from x to y with x pointing to the left - aka North of East)

    :param value: float, angle in COOLEST conventions
    :return: float, angle in lenstronomy conventions
    """
    if value is None:
        return None
    else:
        lenstro_oriented_degree = -value + 90.0
        if lenstro_oriented_degree >= 180.0:
            lenstro_oriented_degree -= 180.0
        elif lenstro_oriented_degree < 0.0:
            lenstro_oriented_degree += 180
        return lenstro_oriented_degree * np.pi / 180.0


def qphi_coolest_to_e1e2_lenstronomy(q, phi):
    """Transform q and phi (axis ratio, position angle East-of-North) to e1,e2 in
    lenstronomy.

    :param q: float, axis ratio
    :param phi: float, position angle in COOLEST conventions
    :return: e1, e2, lenstronomy usual ellipticity parameters
    """
    if None in [q, phi]:
        return None, None
    else:
        angle = degree_coolest_to_radian_lenstronomy(phi)
        e1, e2 = phi_q2_ellipticity(angle, q)
        return e1, e2


def gamma_phi_coolest_to_g1_g2_lenstronomy(gamma_ext, phi_ext):
    """Transform gamma_ext and phi_ext (shear strength, position angle East-of-North) to
    gamma1,gamma2 in lenstronomy.

    :param gamma_ext: float, shear strenght
    :param phi_ext: float, shear angle in COOLEST conventions
    :return: gamma1, gamma2, lenstronomy usual shear parameters
    """
    if None in [gamma_ext, phi_ext]:
        return None, None
    else:
        angle = degree_coolest_to_radian_lenstronomy(phi_ext)
        gamma1, gamma2 = shear_polar2cartesian(angle, gamma_ext)
        return gamma1, gamma2


def ellibounds_coolest_to_lenstronomy(q_down, q_up, phi_down, phi_up):
    """Transforms upper and lower bounds on coolest ellipticity parameters (q, phi)
    towards lenstronomy bound on e1, e2 The mapping can not be perfect but it's the best
    we can do.

    :param q_down: float, lower bound of axis ratio
    :param q_up: float, upper bound of axis ratio
    :param phi_down: float, lower bound of position angle in COOLEST conventions
    :param phi_up: float, upper bound of position angle in COOLEST conventions
    :return: e1_down, e1_up, e2_down, e2_up, bounds for lenstronomy usual ellipticity
        parameters
    """
    if None in [q_down, q_up, phi_down, phi_up]:
        return None, None, None, None
    else:
        e1_down = -(1 - q_down) / (1 + q_down)
        e2_down = -(1 - q_down) / (1 + q_down)

        e1_up = (1 - q_down) / (1 + q_down)
        e2_up = (1 - q_down) / (1 + q_down)

        # WE LOOSE THE ANGLE INFORMATION AND THE OTHER INFO IS NOT WELL TRANSLATED (best we can do)

        return e1_down, e1_up, e2_down, e2_up


def shearbounds_coolest_to_lenstronomy(
    gamma_ext_down, gamma_ext_up, phi_ext_down, phi_ext_up
):
    """Transforms upper and lower bounds on coolest shear parameters (gamma_ext,
    phi_ext) towards lenstronomy bounds on gamma_1, gamma_2 The mapping can not be
    perfect but it's the best we can do.

    :param gamma_ext_down: float, lower bound of shear strenght
    :param gamma_ext_up: float, upper bound of shear strenght
    :param phi_ext_down: float, lower bound of shear position angle in COOLEST
        conventions
    :param phi_ext_up: float, upper bound of shear position angle in COOLEST conventions
    :return: gamma1_down, gamma1_up, gamma2_down, gamma2_up ; bounds for lenstronomy
        usual shear parameters
    """
    if None in [gamma_ext_down, gamma_ext_up, phi_ext_down, phi_ext_up]:
        return None, None, None, None
    else:
        gamma1_down = -gamma_ext_up
        gamma2_down = -gamma_ext_up

        gamma1_up = gamma_ext_up
        gamma2_up = gamma_ext_up

        # WE LOOSE THE ANGLE INFORMATION AND THE OTHER INFO IS NOT WELL TRANSLATED (best we can do)

        return gamma1_down, gamma1_up, gamma2_down, gamma2_up


def update_kwargs_shear(
    shear_idx,
    lens_model_list,
    kwargs_lens,
    kwargs_lens_init,
    kwargs_lens_up,
    kwargs_lens_down,
    kwargs_lens_fixed,
    kwargs_lens_sigma,
    cleaning=False,
):
    """
    Update the lens model list and kwargs with SHEAR mass model (gamma_ext - phi_ext)

    :param shear_idx: coolest.template.classes.profiles.mass.ExternalShear object
    :param lens_model_list: the usual lenstronomy lens_model_list
    :param kwargs_lens: the usual lenstronomy kwargs
    :param kwargs_lens_init: the usual lenstronomy kwargs
    :param kwargs_lens_up: the usual lenstronomy kwargs
    :param kwargs_lens_down: the usual lenstronomy kwargs
    :param kwargs_lens_fixed: the usual lenstronomy kwargs
    :param kwargs_lens_sigma: the usual lenstronomy kwargs
    :param cleaning: bool, if True, will update the empty fields with default values + cleans the kwargs_fixed
    :return: updated list and kwargs
    """
    lens_model_list.append("SHEAR")
    for shear_name, shear_param in shear_idx.parameters.items():
        if shear_name == "gamma_ext":
            gammaext = getattr(shear_param.point_estimate, "value")
            gammaext_up = getattr(shear_param.definition_range, "max_value")
            gammaext_down = getattr(shear_param.definition_range, "min_value")
            gammaext_fixed = gammaext if getattr(shear_param, "fixed") else None
        elif shear_name == "phi_ext":
            psiext = getattr(shear_param.point_estimate, "value")
            psiext_up = getattr(shear_param.definition_range, "max_value")
            psiext_down = getattr(shear_param.definition_range, "min_value")
            psiext_fixed = psiext if getattr(shear_param, "fixed") else None
        else:
            print(f"{shear_name} not known")
    gamma1, gamma2 = gamma_phi_coolest_to_g1_g2_lenstronomy(gammaext, psiext)
    gamma1_fixed, gamma2_fixed = gamma_phi_coolest_to_g1_g2_lenstronomy(
        gammaext_fixed, psiext_fixed
    )
    gamma1_down, gamma1_up, gamma2_down, gamma2_up = shearbounds_coolest_to_lenstronomy(
        gammaext_down, gammaext_up, psiext_down, psiext_up
    )

    kw_1 = {"gamma1": gamma1, "gamma2": gamma2, "ra_0": 0.0, "dec_0": 0.0}
    kw_up_1 = {"gamma1": gamma1_up, "gamma2": gamma2_up, "ra_0": 0.0, "dec_0": 0.0}
    kw_down_1 = {
        "gamma1": gamma1_down,
        "gamma2": gamma2_down,
        "ra_0": 0.0,
        "dec_0": 0.0,
    }
    kw_fixed_1 = {
        "gamma1": gamma1_fixed,
        "gamma2": gamma2_fixed,
        "ra_0": 0.0,
        "dec_0": 0.0,
    }

    kw_ = kw_1.copy()
    kw_init = kw_1.copy()
    kw_up = kw_up_1.copy()
    kw_down = kw_down_1.copy()
    kw_fixed = kw_fixed_1.copy()

    if cleaning is True:
        kw_init_default = {"gamma1": 0.0, "gamma2": -0.0, "ra_0": 0.0, "dec_0": 0.0}
        kw_up_default = {"gamma1": 0.5, "gamma2": 0.5, "ra_0": 100, "dec_0": 100}
        kw_down_default = {"gamma1": -0.5, "gamma2": -0.5, "ra_0": -100, "dec_0": -100}
        for key, val in kw_1.items():
            if val is None:
                kw_init[key] = kw_init_default[key]
        for key, val in kw_up_1.items():
            if val is None:
                kw_up[key] = kw_up_default[key]
        for key, val in kw_down_1.items():
            if val is None:
                kw_down[key] = kw_down_default[key]
        for key, val in kw_fixed_1.items():
            if val is None:
                del kw_fixed[key]

    kwargs_lens.append(kw_)
    kwargs_lens_init.append(kw_init)
    kwargs_lens_up.append(kw_up)
    kwargs_lens_down.append(kw_down)
    kwargs_lens_fixed.append(kw_fixed)
    kwargs_lens_sigma.append({"gamma1": 0.1, "gamma2": 0.1, "ra_0": 0.0, "dec_0": 0.0})
    print("\t Shear correctly added")

    return


def update_kwargs_convergence(
    convergence,
    lens_model_list,
    kwargs_lens,
    kwargs_lens_init,
    kwargs_lens_up,
    kwargs_lens_down,
    kwargs_lens_fixed,
    kwargs_lens_sigma,
    cleaning=False,
):
    """
    Update the lens model list and kwargs with CONVERGENCE mass model (gamma_ext - phi_ext)

    :param profile: coolest.template.classes.profiles.mass.ConvergenceSheet object
    :param lens_model_list: the usual lenstronomy lens_model_list
    :param kwargs_lens: the usual lenstronomy kwargs
    :param kwargs_lens_init: the usual lenstronomy kwargs
    :param kwargs_lens_up: the usual lenstronomy kwargs
    :param kwargs_lens_down: the usual lenstronomy kwargs
    :param kwargs_lens_fixed: the usual lenstronomy kwargs
    :param kwargs_lens_sigma: the usual lenstronomy kwargs
    :param cleaning: bool, if True, will update the empty fields with default values + cleans the kwargs_fixed
    :return: updated list and kwargs
    """
    lens_model_list.append("CONVERGENCE")
    for param_name, param in convergence.parameters.items():
        if param_name == "kappa_s":
            kappa = getattr(param.point_estimate, "value")
            kappa_up = getattr(param.definition_range, "max_value")
            kappa_down = getattr(param.definition_range, "min_value")
            kappa_fixed = kappa if getattr(param, "fixed") else None
        else:
            print(f"{param_name} not known")

    kw_1 = {"kappa": kappa, "ra_0": 0.0, "dec_0": 0.0}
    kw_up_1 = {"kappa": kappa_up, "ra_0": 0.0, "dec_0": 0.0}
    kw_down_1 = {"kappa": kappa_down, "ra_0": 0.0, "dec_0": 0.0}
    kw_fixed_1 = {"kappa": kappa_fixed, "ra_0": 0.0, "dec_0": 0.0}

    kw_ = kw_1.copy()
    kw_init = kw_1.copy()
    kw_up = kw_up_1.copy()
    kw_down = kw_down_1.copy()
    kw_fixed = kw_fixed_1.copy()

    if cleaning is True:
        kw_init_default = {"kappa": 0.0, "ra_0": 0.0, "dec_0": 0.0}
        kw_up_default = {"kappa": 10, "ra_0": 100, "dec_0": 100}
        kw_down_default = {"kappa": -10, "ra_0": -100, "dec_0": -100}
        for key, val in kw_1.items():
            if val is None:
                kw_init[key] = kw_init_default[key]
        for key, val in kw_up_1.items():
            if val is None:
                kw_up[key] = kw_up_default[key]
        for key, val in kw_down_1.items():
            if val is None:
                kw_down[key] = kw_down_default[key]
        for key, val in kw_fixed_1.items():
            if val is None:
                del kw_fixed[key]

    kwargs_lens.append(kw_)
    kwargs_lens_init.append(kw_init)
    kwargs_lens_up.append(kw_up)
    kwargs_lens_down.append(kw_down)
    kwargs_lens_fixed.append(kw_fixed)
    kwargs_lens_sigma.append({"kappa": 0.1, "ra_0": 0.0, "dec_0": 0.0})
    print("\t Convergence correctly added")

    return


def update_kwargs_pemd(
    mass,
    lens_model_list,
    kwargs_lens,
    kwargs_lens_init,
    kwargs_lens_up,
    kwargs_lens_down,
    kwargs_lens_fixed,
    kwargs_lens_sigma,
    cleaning=False,
    use_epl=True,
):
    """Update the lens list and kwargs with PEMD mass model.

    :param mass: coolest.template.classes.profiles.mass.PEMD object
    :param lens_model_list: the usual lenstronomy lens_model_list
    :param kwargs_lens: the usual lenstronomy kwargs
    :param kwargs_lens_init: the usual lenstronomy kwargs
    :param kwargs_lens_up: the usual lenstronomy kwargs
    :param kwargs_lens_down: the usual lenstronomy kwargs
    :param kwargs_lens_fixed: the usual lenstronomy kwargs
    :param kwargs_lens_sigma: the usual lenstronomy kwargs
    :param cleaning: bool, if True, will update the empty fields with default values +
        cleans the kwargs_fixed
    :param use_epl: bool, if True the elliptical power-law profile is 'EPL' instead of
        'PEMD'
    :return: updated list and kwargs
    """
    profile_name = "EPL" if use_epl else "PEMD"
    lens_model_list.append(profile_name)
    for mass_name, mass_param in mass.parameters.items():
        if mass_name == "theta_E":
            te = getattr(mass_param.point_estimate, "value")
            te_up = getattr(mass_param.definition_range, "max_value")
            te_down = getattr(mass_param.definition_range, "min_value")
            te_fixed = te if getattr(mass_param, "fixed") else None
        elif mass_name == "gamma":
            gamma = getattr(mass_param.point_estimate, "value")
            gamma_up = getattr(mass_param.definition_range, "max_value")
            gamma_down = getattr(mass_param.definition_range, "min_value")
            gamma_fixed = gamma if getattr(mass_param, "fixed") else None
        elif mass_name == "q":
            q = getattr(mass_param.point_estimate, "value")
            q_up = getattr(mass_param.definition_range, "max_value")
            q_down = getattr(mass_param.definition_range, "min_value")
            q_fixed = q if getattr(mass_param, "fixed") else None
        elif mass_name == "phi":
            phi = getattr(mass_param.point_estimate, "value")
            phi_up = getattr(mass_param.definition_range, "max_value")
            phi_down = getattr(mass_param.definition_range, "min_value")
            phi_fixed = phi if getattr(mass_param, "fixed") else None
        elif mass_name == "center_x":
            center_x = (
                -getattr(mass_param.point_estimate, "value")
                if getattr(mass_param.point_estimate, "value") is not None
                else None
            )
            center_x_up = getattr(mass_param.definition_range, "max_value")
            center_x_down = getattr(mass_param.definition_range, "min_value")
            center_x_fixed = center_x if getattr(mass_param, "fixed") else None
        elif mass_name == "center_y":
            center_y = getattr(mass_param.point_estimate, "value")
            center_y_up = getattr(mass_param.definition_range, "max_value")
            center_y_down = getattr(mass_param.definition_range, "min_value")
            center_y_fixed = center_y if getattr(mass_param, "fixed") else None
        else:
            print(f"{mass_name} not known")

    e1, e2 = qphi_coolest_to_e1e2_lenstronomy(q, phi)
    e1_fixed, e2_fixed = qphi_coolest_to_e1e2_lenstronomy(q_fixed, phi_fixed)
    e1_down, e1_up, e2_down, e2_up = ellibounds_coolest_to_lenstronomy(
        q_down, q_up, phi_down, phi_up
    )

    kw_1 = {
        "theta_E": te,
        "gamma": gamma,
        "e1": e1,
        "e2": e2,
        "center_x": center_x,
        "center_y": center_y,
    }
    kw_up_1 = {
        "theta_E": te_up,
        "gamma": gamma_up,
        "e1": e1_up,
        "e2": e2_up,
        "center_x": center_x_up,
        "center_y": center_y_up,
    }
    kw_down_1 = {
        "theta_E": te_down,
        "gamma": gamma_down,
        "e1": e1_down,
        "e2": e2_down,
        "center_x": center_x_down,
        "center_y": center_y_down,
    }
    kw_fixed_1 = {
        "theta_E": te_fixed,
        "gamma": gamma_fixed,
        "e1": e1_fixed,
        "e2": e2_fixed,
        "center_x": center_x_fixed,
        "center_y": center_y_fixed,
    }

    kw_ = kw_1.copy()
    kw_init = kw_1.copy()
    kw_up = kw_up_1.copy()
    kw_down = kw_down_1.copy()
    kw_fixed = kw_fixed_1.copy()

    if cleaning is True:
        kw_init_default = {
            "theta_E": 1.0,
            "gamma": 2.0,
            "e1": 0.0,
            "e2": -0.0,
            "center_x": 0.0,
            "center_y": 0.0,
        }
        kw_up_default = {
            "theta_E": 100,
            "gamma": 2.5,
            "e1": 0.5,
            "e2": 0.5,
            "center_x": 100,
            "center_y": 100,
        }
        kw_down_default = {
            "theta_E": 0,
            "gamma": 1.5,
            "e1": -0.5,
            "e2": -0.5,
            "center_x": -100,
            "center_y": -100,
        }
        for key, val in kw_1.items():
            if val is None:
                kw_init[key] = kw_init_default[key]
        for key, val in kw_up_1.items():
            if val is None:
                kw_up[key] = kw_up_default[key]
        for key, val in kw_down_1.items():
            if val is None:
                kw_down[key] = kw_down_default[key]
        for key, val in kw_fixed_1.items():
            if val is None:
                del kw_fixed[key]

    kwargs_lens.append(kw_)
    kwargs_lens_init.append(kw_init)
    kwargs_lens_up.append(kw_up)
    kwargs_lens_down.append(kw_down)
    kwargs_lens_fixed.append(kw_fixed)
    kwargs_lens_sigma.append(
        {
            "theta_E": 1.5,
            "gamma": 0.2,
            "e1": 0.3,
            "e2": 0.3,
            "center_x": 0.5,
            "center_y": 0.5,
        }
    )

    print(f"\t {profile_name} correctly added")

    return


def update_kwargs_sie(
    mass,
    lens_model_list,
    kwargs_lens,
    kwargs_lens_init,
    kwargs_lens_up,
    kwargs_lens_down,
    kwargs_lens_fixed,
    kwargs_lens_sigma,
    cleaning=False,
):
    """Update the lens list and kwargs with SIE mass model.

    :param mass: coolest.template.classes.profiles.mass.SIE object
    :param lens_model_list: the usual lenstronomy lens_model_list
    :param kwargs_lens: the usual lenstronomy kwargs
    :param kwargs_lens_init: the usual lenstronomy kwargs
    :param kwargs_lens_up: the usual lenstronomy kwargs
    :param kwargs_lens_down: the usual lenstronomy kwargs
    :param kwargs_lens_fixed: the usual lenstronomy kwargs
    :param kwargs_lens_sigma: the usual lenstronomy kwargs
    :param cleaning: bool, if True, will update the empty fields with default values +
        cleans the kwargs_fixed
    :return: updated list and kwargs
    """
    lens_model_list.append("SIE")
    for mass_name, mass_param in mass.parameters.items():
        if mass_name == "theta_E":
            te = getattr(mass_param.point_estimate, "value")
            te_up = getattr(mass_param.definition_range, "max_value")
            te_down = getattr(mass_param.definition_range, "min_value")
            te_fixed = te if getattr(mass_param, "fixed") else None
        elif mass_name == "q":
            q = getattr(mass_param.point_estimate, "value")
            q_up = getattr(mass_param.definition_range, "max_value")
            q_down = getattr(mass_param.definition_range, "min_value")
            q_fixed = q if getattr(mass_param, "fixed") else None
        elif mass_name == "phi":
            phi = getattr(mass_param.point_estimate, "value")
            phi_up = getattr(mass_param.definition_range, "max_value")
            phi_down = getattr(mass_param.definition_range, "min_value")
            phi_fixed = phi if getattr(mass_param, "fixed") else None
        elif mass_name == "center_x":
            center_x = (
                -getattr(mass_param.point_estimate, "value")
                if getattr(mass_param.point_estimate, "value") is not None
                else None
            )
            center_x_up = getattr(mass_param.definition_range, "max_value")
            center_x_down = getattr(mass_param.definition_range, "min_value")
            center_x_fixed = center_x if getattr(mass_param, "fixed") else None
        elif mass_name == "center_y":
            center_y = getattr(mass_param.point_estimate, "value")
            center_y_up = getattr(mass_param.definition_range, "max_value")
            center_y_down = getattr(mass_param.definition_range, "min_value")
            center_y_fixed = center_y if getattr(mass_param, "fixed") else None
        else:
            print(f"{mass_name} not known")

    e1, e2 = qphi_coolest_to_e1e2_lenstronomy(q, phi)
    e1_fixed, e2_fixed = qphi_coolest_to_e1e2_lenstronomy(q_fixed, phi_fixed)
    e1_down, e1_up, e2_down, e2_up = ellibounds_coolest_to_lenstronomy(
        q_down, q_up, phi_down, phi_up
    )

    kw_1 = {
        "theta_E": te,
        "e1": e1,
        "e2": e2,
        "center_x": center_x,
        "center_y": center_y,
    }
    kw_up_1 = {
        "theta_E": te_up,
        "e1": e1_up,
        "e2": e2_up,
        "center_x": center_x_up,
        "center_y": center_y_up,
    }
    kw_down_1 = {
        "theta_E": te_down,
        "e1": e1_down,
        "e2": e2_down,
        "center_x": center_x_down,
        "center_y": center_y_down,
    }
    kw_fixed_1 = {
        "theta_E": te_fixed,
        "e1": e1_fixed,
        "e2": e2_fixed,
        "center_x": center_x_fixed,
        "center_y": center_y_fixed,
    }

    kw_ = kw_1.copy()
    kw_init = kw_1.copy()
    kw_up = kw_up_1.copy()
    kw_down = kw_down_1.copy()
    kw_fixed = kw_fixed_1.copy()

    if cleaning is True:
        kw_init_default = {
            "theta_E": 1.0,
            "e1": 0.0,
            "e2": -0.0,
            "center_x": 0.0,
            "center_y": 0.0,
        }
        kw_up_default = {
            "theta_E": 100,
            "e1": 0.5,
            "e2": 0.5,
            "center_x": 100,
            "center_y": 100,
        }
        kw_down_default = {
            "theta_E": 0,
            "e1": -0.5,
            "e2": -0.5,
            "center_x": -100,
            "center_y": -100,
        }
        for key, val in kw_1.items():
            if val is None:
                kw_init[key] = kw_init_default[key]
        for key, val in kw_up_1.items():
            if val is None:
                kw_up[key] = kw_up_default[key]
        for key, val in kw_down_1.items():
            if val is None:
                kw_down[key] = kw_down_default[key]
        for key, val in kw_fixed_1.items():
            if val is None:
                del kw_fixed[key]

    kwargs_lens.append(kw_)
    kwargs_lens_init.append(kw_init)
    kwargs_lens_up.append(kw_up)
    kwargs_lens_down.append(kw_down)
    kwargs_lens_fixed.append(kw_fixed)
    kwargs_lens_sigma.append(
        {"theta_E": 1.5, "e1": 0.3, "e2": 0.3, "center_x": 0.5, "center_y": 0.5}
    )

    print("\t SIE correctly added")

    return


def update_kwargs_sersic(
    light,
    light_model_list,
    kwargs_light,
    kwargs_light_init,
    kwargs_light_up,
    kwargs_light_down,
    kwargs_light_fixed,
    kwargs_light_sigma,
    cleaning=False,
):
    """Update the source list and kwargs with SERSIC_ELLISPE light model.

    :param light: coolest.template.classes.profiles.light.Sersic object
    :param light_model_list: the usual lenstronomy lens_light_model_list or
        source_light_model_list
    :param kwargs_light: the usual lenstronomy kwargs
    :param kwargs_light_init: the usual lenstronomy kwargs
    :param kwargs_light_up: the usual lenstronomy kwargs
    :param kwargs_light_down: the usual lenstronomy kwargs
    :param kwargs_light_fixed: the usual lenstronomy kwargs
    :param kwargs_light_sigma: the usual lenstronomy kwargs
    :param cleaning: bool, if True, will update the empty fields with default values +
        cleans the kwargs_fixed
    :return: updated list and kwargs
    """
    light_model_list.append("SERSIC_ELLIPSE")
    for light_name, light_param in light.parameters.items():
        if light_name == "I_eff":
            amp = getattr(light_param.point_estimate, "value")
            amp_up = getattr(light_param.definition_range, "max_value")
            amp_down = getattr(light_param.definition_range, "min_value")
            amp_fixed = amp if getattr(light_param, "fixed") else None
        elif light_name == "theta_eff":
            R = getattr(light_param.point_estimate, "value")
            R_up = getattr(light_param.definition_range, "max_value")
            R_down = getattr(light_param.definition_range, "min_value")
            R_fixed = R if getattr(light_param, "fixed") else None
        elif light_name == "n":
            n = getattr(light_param.point_estimate, "value")
            n_up = getattr(light_param.definition_range, "max_value")
            n_down = getattr(light_param.definition_range, "min_value")
            n_fixed = n if getattr(light_param, "fixed") else None
        elif light_name == "q":
            q = getattr(light_param.point_estimate, "value")
            q_up = getattr(light_param.definition_range, "max_value")
            q_down = getattr(light_param.definition_range, "min_value")
            q_fixed = q if getattr(light_param, "fixed") else None
        elif light_name == "phi":
            phi = getattr(light_param.point_estimate, "value")
            phi_up = getattr(light_param.definition_range, "max_value")
            phi_down = getattr(light_param.definition_range, "min_value")
            phi_fixed = phi if getattr(light_param, "fixed") else None
        elif light_name == "center_x":
            cx = (
                -getattr(light_param.point_estimate, "value")
                if getattr(light_param.point_estimate, "value") is not None
                else None
            )
            cx_up = getattr(light_param.definition_range, "max_value")
            cx_down = getattr(light_param.definition_range, "min_value")
            cx_fixed = cx if getattr(light_param, "fixed") else None
        elif light_name == "center_y":
            cy = getattr(light_param.point_estimate, "value")
            cy_up = getattr(light_param.definition_range, "max_value")
            cy_down = getattr(light_param.definition_range, "min_value")
            cy_fixed = cy if getattr(light_param, "fixed") else None
        else:
            print(f"Parameter {light_name} unknown in SersicEllipse profile.")

    e1, e2 = qphi_coolest_to_e1e2_lenstronomy(q, phi)
    e1_fixed, e2_fixed = qphi_coolest_to_e1e2_lenstronomy(q_fixed, phi_fixed)
    e1_down, e1_up, e2_down, e2_up = ellibounds_coolest_to_lenstronomy(
        q_down, q_up, phi_down, phi_up
    )

    kw_1 = {
        "amp": amp,
        "R_sersic": R,
        "n_sersic": n,
        "e1": e1,
        "e2": e2,
        "center_x": cx,
        "center_y": cy,
    }
    kw_up_1 = {
        "R_sersic": R_up,
        "n_sersic": n_up,
        "e1": e1_up,
        "e2": e2_up,
        "center_x": cx_up,
        "center_y": cy_up,
    }
    kw_down_1 = {
        "R_sersic": R_down,
        "n_sersic": n_down,
        "e1": e1_down,
        "e2": e2_down,
        "center_x": cx_down,
        "center_y": cy_down,
    }
    kw_fixed_1 = {
        "R_sersic": R_fixed,
        "n_sersic": n_fixed,
        "e1": e1_fixed,
        "e2": e2_fixed,
        "center_x": cx_fixed,
        "center_y": cy_fixed,
    }

    kw_ = kw_1.copy()
    kw_init = kw_1.copy()
    kw_up = kw_up_1.copy()
    kw_down = kw_down_1.copy()
    kw_fixed = kw_fixed_1.copy()

    if cleaning is True:
        kw_init_default = {
            "amp": 1.0,
            "R_sersic": 1.0,
            "n_sersic": 3.5,
            "e1": 0.0,
            "e2": -0.0,
            "center_x": 0.0,
            "center_y": 0.0,
        }
        kw_up_default = {
            "amp": 100,
            "R_sersic": 100,
            "n_sersic": 8,
            "e1": 0.5,
            "e2": 0.5,
            "center_x": 100,
            "center_y": 100,
        }
        kw_down_default = {
            "amp": 0,
            "R_sersic": 0,
            "n_sersic": 0.5,
            "e1": -0.5,
            "e2": -0.5,
            "center_x": -100,
            "center_y": -100,
        }
        for key, val in kw_1.items():
            if val is None:
                kw_init[key] = kw_init_default[key]
        for key, val in kw_up_1.items():
            if val is None:
                kw_up[key] = kw_up_default[key]
        for key, val in kw_down_1.items():
            if val is None:
                kw_down[key] = kw_down_default[key]
        for key, val in kw_fixed_1.items():
            if val is None:
                del kw_fixed[key]

    kwargs_light.append(kw_)
    kwargs_light_init.append(kw_init)
    kwargs_light_up.append(kw_up)
    kwargs_light_down.append(kw_down)
    kwargs_light_fixed.append(kw_fixed)
    kwargs_light_sigma.append(
        {
            "R_sersic": 1.5,
            "n_sersic": 1.5,
            "e1": 0.2,
            "e2": 0.2,
            "center_x": 0.5,
            "center_y": 0.5,
        }
    )
    print("\t Sersic (Ellipse) correctly added")
    return


def update_kwargs_shapelets(
    light,
    light_model_list,
    kwargs_light,
    kwargs_light_init,
    kwargs_light_up,
    kwargs_light_down,
    kwargs_light_fixed,
    kwargs_light_sigma,
    cleaning=False,
):
    """Update the source list and kwargs with SHAPELETS light model.

    :param light: coolest.template.classes.profiles.light.Shapelets object
    :param light_model_list: the usual lenstronomy lens_light_model_list or
        source_light_model_list
    :param kwargs_light: the usual lenstronomy kwargs
    :param kwargs_light_init: the usual lenstronomy kwargs
    :param kwargs_light_up: the usual lenstronomy kwargs
    :param kwargs_light_down: the usual lenstronomy kwargs
    :param kwargs_light_fixed: the usual lenstronomy kwargs
    :param kwargs_light_sigma: the usual lenstronomy kwargs
    :param cleaning: bool, if True, will update the empty fields with default values +
        cleans the kwargs_fixed
    :return: updated list and kwargs
    """
    light_model_list.append("SHAPELETS")
    for light_name, light_param in light.parameters.items():
        if light_name == "beta":
            b = getattr(light_param.point_estimate, "value")
            b_up = getattr(light_param.definition_range, "max_value")
            b_down = getattr(light_param.definition_range, "min_value")
            b_fixed = b if getattr(light_param, "fixed") else None
        elif light_name == "n_max":
            nmax = getattr(light_param.point_estimate, "value")
            nmax_fixed = nmax if getattr(light_param, "fixed") else None
        elif light_name == "center_x":
            cx = (
                -getattr(light_param.point_estimate, "value")
                if getattr(light_param.point_estimate, "value") is not None
                else None
            )
            cx_up = getattr(light_param.definition_range, "max_value")
            cx_down = getattr(light_param.definition_range, "min_value")
            cx_fixed = cx if getattr(light_param, "fixed") else None
        elif light_name == "center_y":
            cy = getattr(light_param.point_estimate, "value")
            cy_up = getattr(light_param.definition_range, "max_value")
            cy_down = getattr(light_param.definition_range, "min_value")
            cy_fixed = cy if getattr(light_param, "fixed") else None
        elif light_name == "amps":
            amp = shapelet_amp_coolest_to_lenstronomy(
                getattr(light_param.point_estimate, "value")
            )
            amp_up = shapelet_amp_coolest_to_lenstronomy(
                getattr(light_param.definition_range, "max_value")
            )
            amp_down = shapelet_amp_coolest_to_lenstronomy(
                getattr(light_param.definition_range, "min_value")
            )
            amp_fixed = amp if getattr(light_param, "fixed") else None
        else:
            print(f"Parameter {light_name} unknown in Shapelets profile.")

    kw_1 = {"amp": amp, "beta": b, "center_x": cx, "center_y": cy, "n_max": nmax}
    kw_up_1 = {"beta": b_up, "center_x": cx_up, "center_y": cy_up}
    kw_down_1 = {"beta": b_down, "center_x": cx_down, "center_y": cy_down}
    kw_fixed_1 = {
        "beta": b_fixed,
        "center_x": cx_fixed,
        "center_y": cy_fixed,
        "n_max": nmax,
    }

    kw_ = kw_1.copy()
    kw_init = kw_1.copy()
    kw_up = kw_up_1.copy()
    kw_down = kw_down_1.copy()
    kw_fixed = kw_fixed_1.copy()

    if cleaning is True:
        kw_init_default = {
            "amp": 1.0,
            "beta": 0.1,
            "center_x": 0.0,
            "center_y": 0.0,
            "n_max": -1,
        }
        kw_up_default = {
            "amp": 100,
            "beta": 100,
            "center_x": 100,
            "center_y": 100,
            "n_max": 1000,
        }
        kw_down_default = {
            "amp": 0,
            "beta": 0,
            "center_x": -100,
            "center_y": -100,
            "n_max": -1,
        }
        for key, val in kw_1.items():
            if val is None:
                kw_init[key] = kw_init_default[key]
        for key, val in kw_up_1.items():
            if val is None:
                kw_up[key] = kw_up_default[key]
        for key, val in kw_down_1.items():
            if val is None:
                kw_down[key] = kw_down_default[key]
        for key, val in kw_fixed_1.items():
            if val is None:
                del kw_fixed[key]

    kwargs_light.append(kw_)
    kwargs_light_init.append(kw_init)
    kwargs_light_up.append(kw_up)
    kwargs_light_down.append(kw_down)
    kwargs_light_fixed.append(kw_fixed)
    kwargs_light_sigma.append({"beta": 0.5, "center_x": 0.5, "center_y": 0.5})
    print("\t Shapelets correctly added")
    return


def update_kwargs_lensed_ps(
    light,
    ps_model_list,
    kwargs_ps,
    kwargs_ps_init,
    kwargs_ps_up,
    kwargs_ps_down,
    kwargs_ps_fixed,
    kwargs_ps_sigma,
    cleaning=False,
):
    """Update the source list and kwargs with lensed point source "LENSED_POSITION"
    light model.

    :param light: coolest.template.classes.profiles.lightLensedPS object
    :param ps_model_list: the usual lenstronomy point_source_model_list
    :param kwargs_ps: the usual lenstronomy kwargs
    :param kwargs_ps_init: the usual lenstronomy kwargs
    :param kwargs_ps_up: the usual lenstronomy kwargs
    :param kwargs_ps_down: the usual lenstronomy kwargs
    :param kwargs_ps_fixed: the usual lenstronomy kwargs
    :param kwargs_ps_sigma: the usual lenstronomy kwargs
    :param cleaning: bool, if True, will update the empty fields with default values +
        cleans the kwargs_fixed
    :return: updated list and kwargs
    """
    ps_model_list.append("LENSED_POSITION")

    try:
        num_ps = len(getattr(light.parameters["ra_list"].point_estimate, "value"))
    except:
        num_ps = 4

    for light_name, light_param in light.parameters.items():
        if light_name == "ra_list":
            ra = (
                -np.array(getattr(light_param.point_estimate, "value"))
                if getattr(light_param.point_estimate, "value") is not None
                else None
            )
            ra_up = getattr(light_param.definition_range, "max_value")
            ra_down = getattr(light_param.definition_range, "min_value")
            ra_fixed = ra if getattr(light_param, "fixed") else None
        elif light_name == "dec_list":
            dec = getattr(light_param.point_estimate, "value")
            dec_up = getattr(light_param.definition_range, "max_value")
            dec_down = getattr(light_param.definition_range, "min_value")
            dec_fixed = dec if getattr(light_param, "fixed") else None
        elif light_name == "amps":
            amp = getattr(light_param.point_estimate, "value")
            amp_up = getattr(light_param.definition_range, "max_value")
            amp_down = getattr(light_param.definition_range, "min_value")
            amp_fixed = amp if getattr(light_param, "fixed") else None
        else:
            print(f"Parameter {light_name} unknown in LensedPS profile.")

    kw_1 = {
        "point_amp": np.array(amp) if amp is not None else None,
        "ra_image": np.array(ra) if ra is not None else None,
        "dec_image": np.array(dec) if dec is not None else None,
    }
    kw_up_1 = {
        "ra_image": np.array(ra_up) if ra_up is not None else None,
        "dec_image": np.array(dec_up) if dec_up is not None else None,
    }
    kw_down_1 = {
        "ra_image": np.array(ra_down) if ra_down is not None else None,
        "dec_image": np.array(dec_down) if dec_down is not None else None,
    }
    kw_fixed_1 = {"ra_image": ra_fixed, "dec_image": dec_fixed}

    kw_ = kw_1.copy()
    kw_init = kw_1.copy()
    kw_up = kw_up_1.copy()
    kw_down = kw_down_1.copy()
    kw_fixed = kw_fixed_1.copy()

    if cleaning is True:
        kw_init_default = {
            "point_amp": np.ones(num_ps),
            "ra_image": np.ones(num_ps),
            "dec_image": np.ones(num_ps),
        }
        kw_up_default = {
            "ra_image": np.ones(num_ps) * 10,
            "dec_image": np.ones(num_ps) * 10,
        }
        kw_down_default = {
            "ra_image": np.ones(num_ps) * (-10),
            "dec_image": np.ones(num_ps) * (-10),
        }
        for key, val in kw_1.items():
            if val is None:
                kw_init[key] = kw_init_default[key]
        for key, val in kw_up_1.items():
            if val is None:
                kw_up[key] = kw_up_default[key]
        for key, val in kw_down_1.items():
            if val is None:
                kw_down[key] = kw_down_default[key]
        for key, val in kw_fixed_1.items():
            if val is None:
                del kw_fixed[key]

    kwargs_ps.append(kw_)
    kwargs_ps_init.append(kw_init)
    kwargs_ps_up.append(kw_up)
    kwargs_ps_down.append(kw_down)
    kwargs_ps_fixed.append(kw_fixed)
    kwargs_ps_sigma.append({"ra_image": np.ones(num_ps), "dec_image": np.ones(num_ps)})
    print("\t Lensed point sources correctly added")
    return
