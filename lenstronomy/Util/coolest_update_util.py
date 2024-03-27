import numpy as np
from coolest.template.classes.parameter import PointEstimate
from coolest.template.classes.probabilities import PosteriorStatistics
from lenstronomy.Util.param_util import ellipticity2phi_q, shear_cartesian2polar


def shapelet_amp_lenstronomy_to_coolest(value):
    """Transforms shapelets coefficients from lenstronomy conventions (x following ra,
    to the left) to COOLEST conventions (x to the right)

    :param value: amplitude of the shapelet (float or np.array) in lenstronomy
        conventions
    :return: amplitude of the shapelet (float or np.array) in COOLEST conventions
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


def radian_lenstronomy_to_degree_coolest(value):
    """
    Transform an angle in radian from lenstronomy (from x to y with x pointing to the left - aka North of East)
    into an angle in degree for the COOLEST (from y to negative x - aka East of North) with folding

    :param value: float, angle in lenstronomy conventions
    :return: float, angle in COOLEST conventions
    """
    coolest_oriented_degree = folding_coolest(radian_lenstronomy_to_degree(value))
    return coolest_oriented_degree


def radian_lenstronomy_to_degree(value):
    """
    Transform an angle in radian from lenstronomy (from x to y with x pointing to the left - aka North of East)
    into an angle in degree for the COOLEST (from y to negative x - aka East of North) without folding in ]-90;90],
    which is COOLEST convention

    :param value: float, angle in lenstronomy conventions
    :return: float, angle almost in COOLEST conventions (without folding)
    """

    lenstro_degree = value * 180.0 / np.pi
    coolest_oriented_degree = lenstro_degree - 90.0
    coolest_oriented_degree *= -1
    return coolest_oriented_degree


def folding_coolest(value):
    """Folds the angle (already in degree with COOLEST East of North convention) into
    COOLEST range, ]-90;90]

    :param value: float, angle almost in COOLEST conventions (without folding in
        ]-90;90])
    :return: float, angle in COOLEST conventions (with folding in ]-90;90])
    """
    coolest_oriented_degree = value
    if type(coolest_oriented_degree) == type(np.array([])):
        for idx, val in enumerate(coolest_oriented_degree):
            if val <= -90.0:
                coolest_oriented_degree[idx] += 180.0
            elif val > 90.0:
                coolest_oriented_degree[idx] -= 180.0
    else:
        if coolest_oriented_degree <= -90:
            coolest_oriented_degree += 180.0
        elif coolest_oriented_degree > 90:
            coolest_oriented_degree -= 180.0
    return coolest_oriented_degree


def e1e2_lenstronomy_to_qphi_coolest(e1, e2):
    """Transform e1,e2 in lenstronomy to q and phi (axis ratio, position angle East-of-
    North)

    :param e1: lenstronomy usual ellipticity parameters
    :param e2: lenstronomy usual ellipticity parameters
    :return: q, phi ; axis ratio and position angle in COOLEST conventions
    """
    angle, q = ellipticity2phi_q(e1, e2)
    phi = radian_lenstronomy_to_degree_coolest(angle)
    return q, phi


def g1g2_lenstronomy_to_gamma_phi_coolest(gamma1, gamma2):
    """Transform gamma1,gamma2 in lenstronomy to gamma_ext and phi_ext (shear strength,
    position angle East-of-North) with folding.

    :param gamma1: lenstronomy usual shear parameters
    :param gamma2: lenstronomy usual shear parameters
    :return: gamma_ext, phi_ext ; shear strenght and shear position angle in COOLEST
        conventions (with folding)
    """
    angle, gamma_ext = shear_cartesian2polar(gamma1, gamma2)
    phi_ext = radian_lenstronomy_to_degree_coolest(angle)
    return gamma_ext, phi_ext


def g1g2_lenstronomy_to_gamma_phi(gamma1, gamma2):
    """Transform gamma1,gamma2 in lenstronomy to gamma_ext and phi_ext (shear strength,
    position angle East-of-North) without folding.

    :param gamma1: lenstronomy usual shear parameters
    :param gamma2: lenstronomy usual shear parameters
    :return: gamma_ext, phi_ext ; shear strenght and shear position angle almost in
        COOLEST conventions (without folding)
    """

    angle = np.arctan2(gamma2, gamma1) / 2.0
    phi_ext = radian_lenstronomy_to_degree(angle)

    gamma_ext = np.sqrt(gamma1**2 + gamma2**2)

    return gamma_ext, phi_ext


def shear_update(shear_idx, kwargs_lens, kwargs_lens_mcmc=None):
    """
    Update the COOLEST SHEAR mass model (gamma_ext - phi_ext) with results in kwargs_lens

    :param shear_idx: coolest.template.classes.profiles.mass.ExternalShear object
    :param kwargs_lens: dictionnary with the point estimate

    :return: updated shear_idx
    """
    gamma_ext, phi_ext = g1g2_lenstronomy_to_gamma_phi_coolest(
        float(kwargs_lens["gamma1"]), float(kwargs_lens["gamma2"])
    )
    shear_idx.parameters["gamma_ext"].set_point_estimate(
        PointEstimate(float(gamma_ext))
    )
    shear_idx.parameters["phi_ext"].set_point_estimate(PointEstimate(float(phi_ext)))

    if kwargs_lens_mcmc is not None:
        g1 = [arg["gamma1"] for arg in kwargs_lens_mcmc]
        g2 = [arg["gamma2"] for arg in kwargs_lens_mcmc]

        g_ext, p_ext = g1g2_lenstronomy_to_gamma_phi(np.array(g1), np.array(g2))

        g_ext_mean = np.mean(g_ext)
        g_ext_16, g_ext_50, g_ext_84 = np.quantile(g_ext, [0.16, 0.5, 0.84])
        shear_idx.parameters["gamma_ext"].set_posterior(
            PosteriorStatistics(
                float(g_ext_mean), float(g_ext_50), float(g_ext_16), float(g_ext_84)
            )
        )

        p_ext_mean = folding_coolest(np.mean(p_ext))
        p_ext_16, p_ext_50, p_ext_84 = folding_coolest(
            np.quantile(p_ext, [0.16, 0.5, 0.84])
        )
        shear_idx.parameters["phi_ext"].set_posterior(
            PosteriorStatistics(
                float(p_ext_mean), float(p_ext_50), float(p_ext_16), float(p_ext_84)
            )
        )
    print("shear correctly updated")

    return


def convergence_update(convergence, kwargs_lens, kwargs_lens_mcmc=None):
    """Update the COOLEST CONVERGENCE mass model (kappa) with results in kwargs_lens.

    :param convergence: coolest.template.classes.profiles.mass.ConvergenceSheet object
    :param kwargs_lens: dictionnary with the point estimate

    :return: updated convergence
    """
    convergence.parameters["kappa_s"].set_point_estimate(
        PointEstimate(float(kwargs_lens["kappa"]))
    )
    if kwargs_lens_mcmc is not None:
        val_samples = [arg["kappa_s"] for arg in kwargs_lens_mcmc]

        val_mean = np.mean(val_samples)
        val_16, val_50, val_84 = np.quantile(val_samples, [0.16, 0.5, 0.84])
        convergence.parameters["kappa_s"].set_posterior(
            PosteriorStatistics(
                float(val_mean), float(val_50), float(val_16), float(val_84)
            )
        )

    print("convergence correctly updated")

    return


def pemd_update(mass, kwargs_lens, kwargs_lens_mcmc=None):
    """Update the COOLEST PEMD mass model with results in kwargs_lens.

    :param mass: coolest.template.classes.profiles.mass.PEMD object
    :param kwargs_lens: dictionnary with the point estimate
    :return: updated mass
    """
    q, phi = e1e2_lenstronomy_to_qphi_coolest(
        float(kwargs_lens["e1"]), float(kwargs_lens["e2"])
    )
    mass.parameters["theta_E"].set_point_estimate(
        PointEstimate(float(kwargs_lens["theta_E"]))
    )
    mass.parameters["gamma"].set_point_estimate(
        PointEstimate(float(kwargs_lens["gamma"]))
    )
    mass.parameters["q"].set_point_estimate(PointEstimate(float(q)))
    mass.parameters["phi"].set_point_estimate(PointEstimate(float(phi)))
    mass.parameters["center_x"].set_point_estimate(
        PointEstimate(-float(kwargs_lens["center_x"]))
    )
    mass.parameters["center_y"].set_point_estimate(
        PointEstimate(float(kwargs_lens["center_y"]))
    )

    if kwargs_lens_mcmc is not None:
        te = [arg["theta_E"] for arg in kwargs_lens_mcmc]
        te_mean = np.mean(te)
        te_16, te_50, te_84 = np.quantile(te, [0.16, 0.5, 0.84])
        mass.parameters["theta_E"].set_posterior(
            PosteriorStatistics(
                float(te_mean), float(te_50), float(te_16), float(te_84)
            )
        )

        g = [arg["gamma"] for arg in kwargs_lens_mcmc]
        g_mean = np.mean(g)
        g_16, g_50, g_84 = np.quantile(g, [0.16, 0.5, 0.84])
        mass.parameters["gamma"].set_posterior(
            PosteriorStatistics(float(g_mean), float(g_50), float(g_16), float(g_84))
        )

        e1 = [arg["e1"] for arg in kwargs_lens_mcmc]
        e2 = [arg["e2"] for arg in kwargs_lens_mcmc]
        ql, phil = e1e2_lenstronomy_to_qphi_coolest(np.array(e1), np.array(e2))

        ql_mean = np.mean(ql)
        ql_16, ql_50, ql_84 = np.quantile(ql, [0.16, 0.5, 0.84])
        mass.parameters["q"].set_posterior(
            PosteriorStatistics(
                float(ql_mean), float(ql_50), float(ql_16), float(ql_84)
            )
        )

        phil_mean = np.mean(phil)
        phil_16, phil_50, phil_84 = np.quantile(phil, [0.16, 0.5, 0.84])
        mass.parameters["phi"].set_posterior(
            PosteriorStatistics(
                float(phil_mean), float(phil_50), float(phil_16), float(phil_84)
            )
        )

        cx = [arg["center_x"] for arg in kwargs_lens_mcmc]
        cx_mean = np.mean(cx)
        cx_16, cx_50, cx_84 = np.quantile(cx, [0.16, 0.5, 0.84])
        mass.parameters["center_x"].set_posterior(
            PosteriorStatistics(
                -float(cx_mean), -float(cx_50), -float(cx_16), -float(cx_84)
            )
        )

        cy = [arg["center_y"] for arg in kwargs_lens_mcmc]
        cy_mean = np.mean(cy)
        cy_16, cy_50, cy_84 = np.quantile(cy, [0.16, 0.5, 0.84])
        mass.parameters["center_y"].set_posterior(
            PosteriorStatistics(
                float(cy_mean), float(cy_50), float(cy_16), float(cy_84)
            )
        )

    print("PEMD correctly updated")

    return


def sie_update(mass, kwargs_lens, kwargs_lens_mcmc=None):
    """Update the COOLEST SIE mass model with results in kwargs_lens.

    :param mass: coolest.template.classes.profiles.mass.SIE object
    :param kwargs_lens : dictionnary with the point estimate
    :return: updated mass
    """

    q, phi = e1e2_lenstronomy_to_qphi_coolest(
        float(kwargs_lens["e1"]), float(kwargs_lens["e2"])
    )
    mass.parameters["theta_E"].set_point_estimate(
        PointEstimate(float(kwargs_lens["theta_E"]))
    )
    mass.parameters["q"].set_point_estimate(PointEstimate(float(q)))
    mass.parameters["phi"].set_point_estimate(PointEstimate(float(phi)))
    mass.parameters["center_x"].set_point_estimate(
        PointEstimate(-float(kwargs_lens["center_x"]))
    )
    mass.parameters["center_y"].set_point_estimate(
        PointEstimate(float(kwargs_lens["center_y"]))
    )

    if kwargs_lens_mcmc is not None:
        te = [arg["theta_E"] for arg in kwargs_lens_mcmc]
        te_mean = np.mean(te)
        te_16, te_50, te_84 = np.quantile(te, [0.16, 0.5, 0.84])
        mass.parameters["theta_E"].set_posterior(
            PosteriorStatistics(
                float(te_mean), float(te_50), float(te_16), float(te_84)
            )
        )

        e1 = [arg["e1"] for arg in kwargs_lens_mcmc]
        e2 = [arg["e2"] for arg in kwargs_lens_mcmc]
        ql, phil = e1e2_lenstronomy_to_qphi_coolest(np.array(e1), np.array(e2))

        ql_mean = np.mean(ql)
        ql_16, ql_50, ql_84 = np.quantile(ql, [0.16, 0.5, 0.84])
        mass.parameters["q"].set_posterior(
            PosteriorStatistics(
                float(ql_mean), float(ql_50), float(ql_16), float(ql_84)
            )
        )

        phil_mean = np.mean(phil)
        phil_16, phil_50, phil_84 = np.quantile(phil, [0.16, 0.5, 0.84])
        mass.parameters["phi"].set_posterior(
            PosteriorStatistics(
                float(phil_mean), float(phil_50), float(phil_16), float(phil_84)
            )
        )

        cx = [arg["center_x"] for arg in kwargs_lens_mcmc]
        cx_mean = np.mean(cx)
        cx_16, cx_50, cx_84 = np.quantile(cx, [0.16, 0.5, 0.84])
        mass.parameters["center_x"].set_posterior(
            PosteriorStatistics(
                -float(cx_mean), -float(cx_50), -float(cx_16), -float(cx_84)
            )
        )

        cy = [arg["center_y"] for arg in kwargs_lens_mcmc]
        cy_mean = np.mean(cy)
        cy_16, cy_50, cy_84 = np.quantile(cy, [0.16, 0.5, 0.84])
        mass.parameters["center_y"].set_posterior(
            PosteriorStatistics(
                float(cy_mean), float(cy_50), float(cy_16), float(cy_84)
            )
        )

    print("SIE correctly updated")

    return


def sersic_update(light, kwargs_light, kwargs_light_mcmc=None):
    """Update the COOLEST Sersic (ellipse) light model with results in kwargs_light.

    :param light: coolest.template.classes.profiles.light.Sersic object
    :param kwargs_light: dictionnary with the point estimate
    :return: updated light
    """
    q, phi = e1e2_lenstronomy_to_qphi_coolest(
        float(kwargs_light["e1"]), float(kwargs_light["e2"])
    )
    light.parameters["I_eff"].set_point_estimate(
        PointEstimate(float(kwargs_light["amp"]))
    )
    light.parameters["theta_eff"].set_point_estimate(
        PointEstimate(float(kwargs_light["R_sersic"]))
    )
    light.parameters["n"].set_point_estimate(
        PointEstimate(float(kwargs_light["n_sersic"]))
    )
    light.parameters["q"].set_point_estimate(PointEstimate(float(q)))
    light.parameters["phi"].set_point_estimate(PointEstimate(float(phi)))
    light.parameters["center_x"].set_point_estimate(
        PointEstimate(-float(kwargs_light["center_x"]))
    )
    light.parameters["center_y"].set_point_estimate(
        PointEstimate(float(kwargs_light["center_y"]))
    )

    if kwargs_light_mcmc is not None:
        a = [arg["amp"] for arg in kwargs_light_mcmc]
        a_mean = np.mean(a)
        a_16, a_50, a_84 = np.quantile(a, [0.16, 0.5, 0.84])
        light.parameters["I_eff"].set_posterior(
            PosteriorStatistics(float(a_mean), float(a_50), float(a_16), float(a_84))
        )

        rs = [arg["R_sersic"] for arg in kwargs_light_mcmc]
        rs_mean = np.mean(rs)
        rs_16, rs_50, rs_84 = np.quantile(rs, [0.16, 0.5, 0.84])
        light.parameters["theta_eff"].set_posterior(
            PosteriorStatistics(
                float(rs_mean), float(rs_50), float(rs_16), float(rs_84)
            )
        )

        ns = [arg["n_sersic"] for arg in kwargs_light_mcmc]
        ns_mean = np.mean(ns)
        ns_16, ns_50, ns_84 = np.quantile(ns, [0.16, 0.5, 0.84])
        light.parameters["n"].set_posterior(
            PosteriorStatistics(
                float(ns_mean), float(ns_50), float(ns_16), float(ns_84)
            )
        )

        e1 = [arg["e1"] for arg in kwargs_light_mcmc]
        e2 = [arg["e2"] for arg in kwargs_light_mcmc]
        ql, phil = e1e2_lenstronomy_to_qphi_coolest(np.array(e1), np.array(e2))

        ql_mean = np.mean(ql)
        ql_16, ql_50, ql_84 = np.quantile(ql, [0.16, 0.5, 0.84])
        light.parameters["q"].set_posterior(
            PosteriorStatistics(
                float(ql_mean), float(ql_50), float(ql_16), float(ql_84)
            )
        )

        phil_mean = np.mean(phil)
        phil_16, phil_50, phil_84 = np.quantile(phil, [0.16, 0.5, 0.84])
        light.parameters["phi"].set_posterior(
            PosteriorStatistics(
                float(phil_mean), float(phil_50), float(phil_16), float(phil_84)
            )
        )

        cx = [arg["center_x"] for arg in kwargs_light_mcmc]
        cx_mean = np.mean(cx)
        cx_16, cx_50, cx_84 = np.quantile(cx, [0.16, 0.5, 0.84])
        light.parameters["center_x"].set_posterior(
            PosteriorStatistics(
                -float(cx_mean), -float(cx_50), -float(cx_16), -float(cx_84)
            )
        )

        cy = [arg["center_y"] for arg in kwargs_light_mcmc]
        cy_mean = np.mean(cy)
        cy_16, cy_50, cy_84 = np.quantile(cy, [0.16, 0.5, 0.84])
        light.parameters["center_y"].set_posterior(
            PosteriorStatistics(
                float(cy_mean), float(cy_50), float(cy_16), float(cy_84)
            )
        )

    print("Sersic (Ellipse) correctly updated")
    return


def shapelets_update(light, kwargs_light, kwargs_light_mcmc=None):
    """Update the COOLEST Shapelets light model with results in kwargs_light.

    :param light: coolest.template.classes.profiles.light.Shapelets object
    :param kwargs_light: dictionnary with the point estimate
    :return: updated light
    """
    light.parameters["amps"].set_point_estimate(
        PointEstimate(
            shapelet_amp_lenstronomy_to_coolest(np.ndarray.tolist(kwargs_light["amp"]))
        )
    )
    light.parameters["beta"].set_point_estimate(
        PointEstimate(float(kwargs_light["beta"]))
    )
    light.parameters["n_max"].set_point_estimate(
        PointEstimate(int(kwargs_light["n_max"]))
    )
    light.parameters["center_x"].set_point_estimate(
        PointEstimate(-float(kwargs_light["center_x"]))
    )
    light.parameters["center_y"].set_point_estimate(
        PointEstimate(float(kwargs_light["center_y"]))
    )

    if kwargs_light_mcmc is not None:
        a = [arg["amp"] for arg in kwargs_light_mcmc]
        a_mean = np.mean(a, axis=0)
        a_16, a_50, a_84 = np.quantile(a, [0.16, 0.5, 0.84], axis=0)
        light.parameters["amps"].set_posterior(
            PosteriorStatistics(
                shapelet_amp_lenstronomy_to_coolest(np.ndarray.tolist(a_mean)),
                shapelet_amp_lenstronomy_to_coolest(np.ndarray.tolist(a_50)),
                shapelet_amp_lenstronomy_to_coolest(np.ndarray.tolist(a_16)),
                shapelet_amp_lenstronomy_to_coolest(np.ndarray.tolist(a_84)),
            )
        )

        b = [arg["beta"] for arg in kwargs_light_mcmc]
        b_mean = np.mean(b)
        b_16, b_50, b_84 = np.quantile(b, [0.16, 0.5, 0.84])
        light.parameters["beta"].set_posterior(
            PosteriorStatistics(float(b_mean), float(b_50), float(b_16), float(b_84))
        )

        nmax = [arg["n_max"] for arg in kwargs_light_mcmc]
        nmax_mean = np.mean(nmax)
        nmax_16, nmax_50, nmax_84 = np.quantile(nmax, [0.16, 0.5, 0.84])
        light.parameters["n_max"].set_posterior(
            PosteriorStatistics(
                float(nmax_mean), float(nmax_50), float(nmax_16), float(nmax_84)
            )
        )

        cx = [arg["center_x"] for arg in kwargs_light_mcmc]
        cx_mean = np.mean(cx)
        cx_16, cx_50, cx_84 = np.quantile(cx, [0.16, 0.5, 0.84])
        light.parameters["center_x"].set_posterior(
            PosteriorStatistics(
                -float(cx_mean), -float(cx_50), -float(cx_16), -float(cx_84)
            )
        )

        cy = [arg["center_y"] for arg in kwargs_light_mcmc]
        cy_mean = np.mean(cy)
        cy_16, cy_50, cy_84 = np.quantile(cy, [0.16, 0.5, 0.84])
        light.parameters["center_y"].set_posterior(
            PosteriorStatistics(
                float(cy_mean), float(cy_50), float(cy_16), float(cy_84)
            )
        )

    print("Shapelets correctly updated")
    return


def lensed_point_source_update(light, kwargs_ps, kwargs_ps_mcmc=None):
    """Update the COOLEST LensedPS light model with results in kwargs_ps.

    :param light: coolest.template.classes.profiles.light.LensedPS object
    :param kwargs_ps: dictionnary with the point estimate
    :return: updated light
    """
    light.parameters["amps"].set_point_estimate(
        PointEstimate(np.ndarray.tolist(kwargs_ps["point_amp"]))
    )
    light.parameters["ra_list"].set_point_estimate(
        PointEstimate(np.ndarray.tolist(-kwargs_ps["ra_image"]))
    )
    light.parameters["dec_list"].set_point_estimate(
        PointEstimate(np.ndarray.tolist(kwargs_ps["dec_image"]))
    )

    if kwargs_ps_mcmc is not None:
        a = [arg["point_amp"] for arg in kwargs_ps_mcmc]
        a_mean = np.mean(a, axis=0)
        a_16, a_50, a_84 = np.quantile(a, [0.16, 0.5, 0.84], axis=0)
        light.parameters["amps"].set_posterior(
            PosteriorStatistics(
                np.ndarray.tolist(a_mean),
                np.ndarray.tolist(a_50),
                np.ndarray.tolist(a_16),
                np.ndarray.tolist(a_84),
            )
        )

        ra = [arg["ra_image"] for arg in kwargs_ps_mcmc]
        ra_mean = np.mean(ra, axis=0)
        ra_16, ra_50, ra_84 = np.quantile(ra, [0.16, 0.5, 0.84], axis=0)
        light.parameters["ra_list"].set_posterior(
            PosteriorStatistics(
                np.ndarray.tolist(-ra_mean),
                np.ndarray.tolist(-ra_50),
                np.ndarray.tolist(-ra_16),
                np.ndarray.tolist(-ra_84),
            )
        )
        dec = [arg["dec_image"] for arg in kwargs_ps_mcmc]
        dec_mean = np.mean(dec, axis=0)
        dec_16, dec_50, dec_84 = np.quantile(dec, [0.16, 0.5, 0.84], axis=0)
        light.parameters["dec_list"].set_posterior(
            PosteriorStatistics(
                np.ndarray.tolist(dec_mean),
                np.ndarray.tolist(dec_50),
                np.ndarray.tolist(dec_16),
                np.ndarray.tolist(dec_84),
            )
        )

    print("Lensed point source correctly updated")
    return
