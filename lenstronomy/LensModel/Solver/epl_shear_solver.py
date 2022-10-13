__author__ = 'ewoudwempe', 'sibirrer'

import numpy as np
from lenstronomy.LensModel.Util.epl_util import min_approx, pol_to_cart, cart_to_pol, cdot, ps, rotmat, solvequadeq, brentq_inline
from lenstronomy.LensModel.Util.epl_util import pol_to_ell, ell_to_pol, geomlinspace
from lenstronomy.Util.image_util import findOverlap
from lenstronomy.LensModel.Profiles.epl_numba import alpha, omega
from lenstronomy.Util.numba_util import jit
from lenstronomy.Util.param_util import ellipticity2phi_q, shear_cartesian2polar, shear_polar2cartesian
from lenstronomy.LensModel.Profiles.shear import Shear


@jit()
def _alpha_epl_shear(x, y, b, q, t=1, gamma1=0, gamma2=0, Omega=None):
    """The complex deflection of EPL+SHEAR"""
    return alpha(x, y, b, q, t=t, Omega=Omega) + (gamma1 * x + gamma2*y) + 1j * (gamma2*x - gamma1*y)


@jit()
def _one_dim_lens_eq_calcs(args, phi):
    """Calculates intermediate quantities that are needed for several of the subsequent functions"""
    b, t, y1, y2, q, gamma1, gamma2 = args
    y = (y1+y2*1j)

    rhat = 1/(1-gamma1**2-gamma2**2)*(
            ((1+gamma1)*np.cos(phi)+gamma2*np.sin(phi))+1j*(gamma2*np.cos(phi)+(1-gamma1)*np.sin(phi)))
    thetahat = 1/(1-gamma1**2-gamma2**2)*(
            ((1+gamma1)*np.sin(phi)-gamma2*np.cos(phi))+1j*(gamma2*np.sin(phi)-(1-gamma1)*np.cos(phi)))

    frac_Roverrsh, phiell = pol_to_ell(1, phi, q)
    Omega = omega(phiell, t, q)
    const = 2*b/(1+q)
    if abs(t-1) > 1e-4:
        b_over_r_pow_tm1 = -cdot(y, thetahat)/(const*cdot(Omega, thetahat))
        R = b*np.abs(b_over_r_pow_tm1)**(1/(1-t))*np.sign(b_over_r_pow_tm1)
    else:
        Omega_ort = 1j*Omega
        x = ((1-gamma1)*np.cos(phi)-gamma2*np.sin(phi))+1j*(-gamma2*np.cos(phi)+(1+gamma1)*np.sin(phi))
        R = cdot(Omega_ort, y)/cdot(Omega_ort, x)*frac_Roverrsh
    r, theta = ell_to_pol(R, phiell, q)
    return Omega, const, phiell, q, r, rhat, t, b, thetahat, y


@jit()
def _one_dim_lens_eq_both(phi, args):
    """Calculates and returns simultaneously both the smooth and the not-smooth 1-dimensional lens equation
    that needs to be solved"""
    Omega, const, phiell, q, r, rhat, t, b, thetahat, y = _one_dim_lens_eq_calcs(args, phi)
    rr, thetaa = ell_to_pol(1, phiell, q)
    ip = cdot(y, rhat)*cdot(Omega, thetahat)-cdot(Omega, rhat)*cdot(y, thetahat)
    # The derivations are lost somewhere in my notes...
    eq = (rr*b)**(2/t-2)*ps((cdot(y, thetahat)/const), 2/t)*ip**2+ps(ip, 2/t)*cdot(Omega, thetahat)**2
    eq_notsmooth = ps(rr*b, 1-t)*(cdot(y, thetahat)/const)*np.abs(ip)**t+ip*np.abs(cdot(Omega, thetahat))**(+t)
    return eq, eq_notsmooth


@jit()
def _getr(phi, args):
    """Given an angle phi, get the radius r"""
    Omega, const, phiell, q, r, rhat, t, b, thetahat, y = _one_dim_lens_eq_calcs(args, phi)
    return r


@jit()
def _one_dim_lens_eq(phi, args):
    """Calculates the smooth 1-dimensional lens equation to solve - to be used by a root-finder"""
    Omega, const, phiell, q, r, rhat, t, b, thetahat, y = _one_dim_lens_eq_calcs(args, phi)

    rr, thetaa = ell_to_pol(1, phiell, q)
    ip = cdot(y, rhat)*cdot(Omega, thetahat)-cdot(Omega, rhat)*cdot(y, thetahat)
    eq = (rr*b)**(2/t-2)*ps((cdot(y, thetahat)/const), 2/t)*ip**2+ps(ip, 2/t)*cdot(Omega, thetahat)**2
    return eq


@jit()
def _one_dim_lens_eq_unsmooth(phi, args):
    """Calculates the not-smooth 1-dimensional lens equation to to solve - to be used by a root-finder.
    For some parameters and solutions, numerical issues make solving this one feasible while the other is not."""
    Omega, const, phiell, q, r, rhat, t, b, thetahat, y = _one_dim_lens_eq_calcs(args, phi)

    rr, thetaa = ell_to_pol(1, phiell, q)
    ip = cdot(y, rhat)*cdot(Omega, thetahat)-cdot(Omega, rhat)*cdot(y, thetahat)
    eq_notsmooth = ps(rr*b, 1-t)*(cdot(y, thetahat)/const)*np.abs(ip)**t+ip*np.abs(cdot(Omega, thetahat))**(+t)
    return eq_notsmooth


@jit()
def _getphi(thpl, args):
    """
    Finds all roots to both versions the 1-dimensional lens equation in phi, by doing a grid search for sign changes on
    the supplied thpl. In the case of extrema, refine at the relevant location.

    :param thpl: What points to calculate the equation on use for detecting sign changes
    :param args: Parameters to be passed to the lens equation
    :return: an array containing all roots
    """
    y, y_notsmooth = _one_dim_lens_eq_both(thpl, args)
    num_phi = len(thpl)
    roots = []
    for i in range(num_phi-1):
        if y[i+1]*y[i] <= 0:
            roots.append(brentq_inline(_one_dim_lens_eq, thpl[i], thpl[i+1], args=args) % (2*np.pi))
        elif y_notsmooth[i+1]*y_notsmooth[i] <= 0:
            roots.append(brentq_inline(_one_dim_lens_eq_unsmooth, thpl[i], thpl[i+1], args=args) % (2*np.pi))

    for i in range(1, num_phi+1):
        y1 = y[i-1]
        y2 = y[i % num_phi]
        y3 = y[(i+1) % num_phi]
        y1n = y_notsmooth[(i-1) % num_phi]
        y2n = y_notsmooth[i % num_phi]
        y3n = y_notsmooth[(i+1) % num_phi]
        if (y3-y2)*(y2-y1) <= 0 or (y3n-y2n)*(y2n-y1n) <= 0:
            if y3*y2 <= 0 or y1*y2 <= 0:
                continue
            if i > num_phi-2:
                continue
            else:
                x1 = thpl[i - 1]
                x2 = thpl[i]
                x3 = thpl[(i + 1)]

            xmin = min_approx(x1, x2, x3, y1, y2, y3)
            xmin_ns = min_approx(x1, x2, x3, y1n, y2n, y3n)
            ymin = _one_dim_lens_eq(xmin, args)
            ymin_ns = _one_dim_lens_eq_unsmooth(xmin_ns, args)
            if ymin*y2 <= 0 and x2 <= xmin <= x3:
                roots.append(brentq_inline(_one_dim_lens_eq, x2, xmin, args=args) % (2*np.pi))
                roots.append(brentq_inline(_one_dim_lens_eq, xmin, x3, args=args) % (2*np.pi))
            elif ymin*y2 <= 0 and x1 <= xmin <= x2:
                roots.append(brentq_inline(_one_dim_lens_eq, x1, xmin, args=args) % (2*np.pi))
                roots.append(brentq_inline(_one_dim_lens_eq, xmin, x2, args=args) % (2*np.pi))
            elif ymin_ns * y2n <= 0 and x2 <= xmin_ns <= x3:
                roots.append(brentq_inline(_one_dim_lens_eq_unsmooth, x2, xmin_ns, args=args) % (2*np.pi))
                roots.append(brentq_inline(_one_dim_lens_eq_unsmooth, xmin_ns, x3, args=args) % (2*np.pi))
            elif ymin_ns*y2n <= 0 and x1 <= xmin_ns <= x2:
                roots.append(brentq_inline(_one_dim_lens_eq_unsmooth, x1, xmin_ns, args=args) % (2*np.pi))
                roots.append(brentq_inline(_one_dim_lens_eq_unsmooth, xmin_ns, x2, args=args) % (2*np.pi))

    return np.array(roots)


def solvelenseq_majoraxis(args, Nmeas=200, Nmeas_extra=50):
    """Solve the lens equation, where the arguments have been properly rotated to the major-axis"""
    b, t, y1, y2, q, gamma1, gamma2 = args
    p1 = np.arctan2(y2*(1-gamma1)+gamma2*y1, y1*(1+gamma1)+gamma2*y2)
    int_points = [p1]
    geom = geomlinspace(1e-4, 0.1, Nmeas_extra)

    thpl = np.sort(np.concatenate((np.linspace(0., np.pi, Nmeas),
                                   *[i % np.pi-geom for i in int_points],
                                   *[i % np.pi+geom for i in int_points],
                                   )))
    the = _getphi(thpl, (b, t, y1, y2, q, gamma1, gamma2))
    thetas = np.concatenate((the, the + np.pi))
    Rs = np.array([_getr(theta, (b, t, y1, y2, q, gamma1, gamma2)) for theta in thetas])
    stuff = np.array(pol_to_cart(Rs[Rs > 0], thetas[Rs > 0]))
    diff = -y1-y2*1j+stuff[0]+stuff[1]*1j-_alpha_epl_shear(stuff[0], stuff[1], b, q, t, gamma1=gamma1, gamma2=gamma2)
    goodones = np.abs(diff) < 1e-8
    return findOverlap(*stuff[:, goodones], 1e-8)


def _check_center(kwargs_lens):
    """Checks if the shear-at-center convention is properly used."""
    # calculate (inverse) displacement caused by the offset between shear and lens centroid
    # this shift needs to be added to the source position such that the solution of the lens equation
    # without this shift in the shear is the correct one
    if len(kwargs_lens) > 1:
        shear = Shear()
        # calculate shift from the deflector centroid from the shear field
        alpha_x, alpha_y = shear.derivatives(kwargs_lens[0]['center_x'], kwargs_lens[0]['center_y'], **kwargs_lens[1])
        return alpha_x, alpha_y
    else:
        return 0, 0


def solve_lenseq_pemd(pos_, kwargs_lens, Nmeas=400, Nmeas_extra=80, **kwargs):
    """
    Solves the lens equation using a semi-analytical recipe.
    :param pos_: The source plane position (shape (2,)), or the source plane positions (shape (2,N)) for which to solve the lens equation 
    :param kwargs_lens: List of kwargs in lenstronomy style, following ['EPL', 'SHEAR'] format
    :param Nmeas: resolution with which to sample the angular grid, higher means more reliable lens equation solving. For solving many positions at once, you may want to set this higher.
    :param Nmeas_extra: resolution with which to additionally sample the angular grid at the low-shear end, higher means more reliable lens equation solving. For solving many positions at once, you may want to set this higher.
    :return: The lens plane positions.
    Note: generally the (demagnified) central image will also be included.
    """
    pos = np.asarray(pos_)
    if pos.ndim > 1 and pos.shape[-1] != 1:
        pos = pos[..., None]
    t = kwargs_lens[0]['gamma']-1 if 'gamma' in kwargs_lens[0] else 1

    theta_ell, q = ellipticity2phi_q(kwargs_lens[0]['e1'], kwargs_lens[0]['e2'])
    b = kwargs_lens[0]['theta_E']*np.sqrt(q)
    if len(kwargs_lens) > 1:
        gamma = kwargs_lens[1]['gamma1']+1j*kwargs_lens[1]['gamma2']
    else:
        gamma = 0+0j
    shift_x, shift_y = _check_center(kwargs_lens)
    shift = shift_x + 1j * shift_y
    cen = kwargs_lens[0]['center_x']+1j*kwargs_lens[0]['center_y']
    p = pos[0]+1j*pos[1] - cen + shift

    rotfact = np.exp(-1j*theta_ell)
    gamma *= rotfact**2
    p *= rotfact
    res = solvelenseq_majoraxis((b, t, p.real, p.imag, q,
                                 gamma.real, gamma.imag), Nmeas=Nmeas, Nmeas_extra=Nmeas_extra)
    xsol, ysol = res
    x = np.array([(xs+1j*ys)/rotfact+cen for xs, ys in zip(xsol, ysol)])
    return x.real, x.imag


def caustics_epl_shear(kwargs_lens, num_th=500, maginf=0, sourceplane=True, return_which=None):
    """
    Analytically calculates the caustics of an EPL+shear lens model.
    Since for gamma>2, the outer critical curve does not exist, the option to find the curves for a set, finite magnification exists, by supplying maginf, so that the routine finds the curve of this magnification, rather than the true caustic.

    :param kwargs_lens: List of kwargs in lenstronomy style, following ['EPL', 'SHEAR'] format
    :param num_th: resolution.
    :param maginf: the outer critical curve for t>1 will be replaced with the curve where the inverse magnification is maginf
    :param sourceplane: if True (default), ray-shoot the calculated critical curves to the source plane
    :param return_which: options 'quad' (boundary of area within which there are 4 images), 'double' (boundary of area within which there are 2 images),
     'caustic' (the diamond caustic) and 'cut' (the cut, if it exists, that is if t<2, else, if t>2, returns the caustic) and None (in that case: return quad, caustic, cut)
    :return: (2,N) array if return_which set, else a tuple of (caustic, cut, quad)
    """
    e1, e2 = kwargs_lens[0]['e1'], kwargs_lens[0]['e2']
    if len(kwargs_lens) > 1:
        gamma1unr, gamma2unr = kwargs_lens[1]['gamma1'], kwargs_lens[1]['gamma2']
    else:
        gamma1unr, gamma2unr = 0, 0
    t = kwargs_lens[0]['gamma']-1 if 'gamma' in kwargs_lens[0] else 1
    theta_ell, q = ellipticity2phi_q(e1, e2)
    theta_gamma, gamma_mag = shear_cartesian2polar(gamma1unr, gamma2unr)
    b = np.sqrt(q)*kwargs_lens[0]['theta_E']
    cen = np.expand_dims(np.array([kwargs_lens[0]['center_x'], kwargs_lens[0]['center_y']]), 1)
    theta_gamma -= theta_ell
    gamma1, gamma2 = shear_polar2cartesian(theta_gamma, gamma_mag)
    M = rotmat(-theta_ell)
    phiell, q = ellipticity2phi_q(e1, e2)
    theta = np.linspace(0, 2*np.pi, num_th, endpoint=False)
    r = 1
    R, phi = pol_to_ell(1, theta, q)
    Omega = omega(phi, t, q)
    aa = 1
    bb = -(2-t)
    frac_roverR = r/R
    cc = (1-t)*(2-t)*(cdot(np.exp(1j*theta), Omega))/frac_roverR*2/(1+q)
    cc -= (1-t)**2*(2/(1+q))**2*np.abs(Omega)**2/frac_roverR**2
    # Shear stuff:
    gammaint_fac = (-np.exp(2j*theta)*(2-t)/2+(1-t)*np.exp(1j*theta)*2/(1+q)*Omega/frac_roverR)
    gamma = gamma1+1j*gamma2
    aa -= np.abs(gamma)**2
    bb -= 2*cdot(gamma, gammaint_fac)
    usol = np.array(solvequadeq(cc, bb, aa)).T
    xcr_4, ycr_4 = pol_to_cart(b*usol[:, 1]**(-1/t)*frac_roverR, theta)
    if t > 1:  # If t>1, get the approximate outer caustic instead (where inverse magnification = maginf).
        usol = np.array(solvequadeq(cc, bb, aa-maginf)).T
        xcr_cut, ycr_cut = pol_to_cart(b*usol[:, 1]**(-1/t)*frac_roverR, theta)
    else:
        usol = np.array(solvequadeq(cc, bb, aa+maginf)).T
        xcr_cut, ycr_cut = pol_to_cart(b*usol[:, 0]**(-1/t)*frac_roverR, theta)
    al_cut = _alpha_epl_shear(xcr_cut, ycr_cut, b, q, t, gamma1, gamma2, Omega=Omega)
    al_4 = _alpha_epl_shear(xcr_4, ycr_4, b, q, t, gamma1, gamma2, Omega=Omega)
    if sourceplane:
        xca_cut, yca_cut = xcr_cut - al_cut.real, ycr_cut - al_cut.imag
        xca_4, yca_4 = xcr_4 - al_4.real, ycr_4 - al_4.imag
    else:
        xca_cut, yca_cut = xcr_cut, ycr_cut
        xca_4, yca_4 = xcr_4, ycr_4
    if return_which == 'caustic':
        return M@(xca_4, yca_4) + cen
    if return_which == 'cut':
        return M@(xca_cut, yca_cut) + cen

    rcut, thcut = cart_to_pol(xca_cut, yca_cut)
    r, th = cart_to_pol(xca_4, yca_4)
    r2 = np.interp(th, thcut, rcut, period=2*np.pi)

    if return_which == 'double':
        r = np.fmax(r, r2)
    else:  # Quad
        r = np.fmin(r, r2)

    pos_tosample = np.empty((2, num_th))
    pos_tosample[0], pos_tosample[1] = pol_to_cart(r, th)
    if return_which in ('double', 'quad'):
        return M@pos_tosample + cen

    return M@(xca_4, yca_4) + cen, M@(xca_cut, yca_cut) + cen, M@pos_tosample + cen  # Mostly for some backward compatibility
