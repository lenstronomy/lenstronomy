__author__ = "rgh, Wei Du"

import lenstronomy.Util.util as util
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.spp import SPP
import numpy as np
from scipy.integrate import quad
from scipy.special import hyp2f1,spence,beta
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


__all__ = ["BPL", "BPLMajorAxis"]


class BPL(LensProfileBase):
    """Elliptical Power Law mass profile.

    .. math::
        \\kappa(x, y) = \\frac{3-\\gamma}{2} \\left(\\frac{\\theta_{E}}{\\sqrt{q x^2 + y^2/q}} \\right)^{\\gamma-1}

    with :math:`\\theta_{E}` is the (circularized) Einstein radius,
    :math:`\\gamma` is the negative power-law slope of the 3D mass distributions,
    :math:`q` is the minor/major axis ratio,
    and :math:`x` and :math:`y` are defined in a coordinate system aligned with the major and minor axis of the lens.

    In terms of eccentricities, this profile is defined as

    .. math::
        \\kappa(r) = \\frac{3-\\gamma}{2} \\left(\\frac{\\theta'_{E}}{r \\sqrt{1 - e*\\cos(2*\\phi)}} \\right)^{\\gamma-1}

    with :math:`\\epsilon` is the ellipticity defined as

    .. math::
        \\epsilon = \\frac{1-q^2}{1+q^2}

    And an Einstein radius :math:`\\theta'_{\\rm E}` related to the definition used is

    .. math::
        \\left(\\frac{\\theta'_{\\rm E}}{\\theta_{\\rm E}}\\right)^{2} = \\frac{2q}{1+q^2}.

    The mathematical form of the calculation is presented by Tessore & Metcalf (2015), https://arxiv.org/abs/1507.01819.
    The current implementation is using hyperbolic functions. The paper presents an iterative calculation scheme,
    converging in few iterations to high precision and accuracy.

    A (faster) implementation of the same model using numba is accessible as 'EPL_NUMBA' with the iterative calculation
    scheme. An alternative implementation of the same model using a fortran code FASTELL is implemented as 'PEMD'
    profile.
    """
    # b, a, a_c, r_c,q
    param_names = ["b", "a", "a_c", "r_c", "e1", "e2", "center_x", "center_y"]
    lower_limit_default = {
        "b": 0,
        "a": 1,
        "a_c": 0,
        "r_c": 0,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "b": 100,
        "a": 3,
        "a_c": 3,
        "r_c": 100,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(self):
        self.bpl_major_axis = BPLMajorAxis()
        #self.spp = SPP()
        super(BPL, self).__init__()

    def param_conv(self, b, a, a_c, r_c, e1, e2):
        """Converts parameters as defined in this class to the parameters used in the
        EPLMajorAxis() class.

        :param theta_E: Einstein radius as defined in the profile class
        :param gamma: negative power-law slope
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :return: b, t, q, phi_G
        """
        if self._static is True:
            return self._b_static, self._a_static,self._a_c_static, self._r_c_static, self._q_static, self._phi_G_static
        return self._param_conv(b, a, a_c, r_c, e1, e2)

    @staticmethod
    def _param_conv(b, a, a_c, r_c, e1, e2):
        """Convert parameters from :math:`R = \\sqrt{q x^2 + y^2/q}` to :math:`R =
        \\sqrt{q^2 x^2 + y^2}`

        :param gamma: power law slope
        :param theta_E: Einstein radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :return: critical radius b, slope t, axis ratio q, orientation angle phi_G
        """
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        return  b, a, a_c, r_c, q, phi_G

    def set_static(self, b, a, a_c, r_c, e1, e2, center_x=0, center_y=0):
        """

        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: self variables set
        """
        self._static = True
        (
            self._b_static,
            self._a_static,
            self._a_c_static,
            self._r_c_static,
            self._q_static,
            self._phi_G_static,
        ) = self._param_conv(b, a, a_c, r_c, e1, e2)

    def set_dynamic(self):
        """

        :return:
        """
        self._static = False
        if hasattr(self, "_b_static"):
            del self._b_static
        if hasattr(self, "_a_static"):
            del self._a_static
        if hasattr(self, "_a_c_static"):
            del self._a_c_static
        if hasattr(self, "_r_c_static"):
            del self._r_c_static
        if hasattr(self, "_phi_G_static"):
            del self._phi_G_static
        if hasattr(self, "_q_static"):
            del self._q_static

    def function(self, x, y, b, a, a_c, r_c, e1, e2, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: lensing potential
        """
        b, a, a_c, r_c, q, phi_G = self.param_conv(b, a, a_c, r_c, e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f_ = self.bpl_major_axis.function(x__, y__, b, a, a_c, r_c, q)
        # rotate back
        return f_

    def derivatives(self, x, y, b, a, a_c, r_c, e1, e2, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: alpha_x, alpha_y
        """
        b, a, a_c, r_c, q, phi_G = self.param_conv(b, a, a_c, r_c, e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        #print(np.max(x_)-np.min(x_),np.sqrt(x_.shape[0]),x_)
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f__x, f__y = self.bpl_major_axis.derivatives(x__, y__, b, a, a_c, r_c,q)
        # rotate back
        f_x, f_y = util.rotate(f__x, f__y, -phi_G)
        return f_x, f_y

    def hessian(self, x, y, b, a, a_c, r_c, e1, e2, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: f_xx, f_xy, f_yx, f_yy
        """

        b, a, a_c, r_c, q, phi_G = self.param_conv(b, a, a_c, r_c, e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f__xx, f__xy, f__yx, f__yy = self.bpl_major_axis.hessian(x__, y__,b, a, a_c, r_c, q)
        # rotate back
        kappa = 1.0 / 2 * (f__xx + f__yy)
        gamma1__ = 1.0 / 2 * (f__xx - f__yy)
        gamma2__ = f__xy
        gamma1 = np.cos(2 * phi_G) * gamma1__ - np.sin(2 * phi_G) * gamma2__
        gamma2 = +np.sin(2 * phi_G) * gamma1__ + np.cos(2 * phi_G) * gamma2__
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy

    #def mass_3d_lens(self, r, theta_E, gamma, e1=None, e2=None):
        """Computes the spherical power-law mass enclosed (with SPP routine)

        :param r: radius within the mass is computed
        :param theta_E: Einstein radius
        :param gamma: power-law slope
        :param e1: eccentricity component (not used)
        :param e2: eccentricity component (not used)
        :return: mass enclosed a 3D radius r.
        """
    #    return self.spp.mass_3d_lens(r, theta_E, gamma)

    #def density_lens(self, r, theta_E, gamma, e1=None, e2=None):
        """Computes the density at 3d radius r given lens model parameterization. The
        integral in the LOS projection of this quantity results in the convergence
        quantity.

        :param r: radius within the mass is computed
        :param theta_E: Einstein radius
        :param gamma: power-law slope
        :param e1: eccentricity component (not used)
        :param e2: eccentricity component (not used)
        :return: mass enclosed a 3D radius r
        """
    #    return self.spp.density_lens(r, theta_E, gamma)


class BPLMajorAxis(LensProfileBase):
    """This class contains the function and the derivatives of the elliptical power law.

    .. math::
        \\kappa = (2-t)/2 * \\left[\\frac{b}{\\sqrt{q^2 x^2 + y^2}}\\right]^t

    where with :math:`t = \\gamma - 1` (from EPL class) being the projected power-law slope of the convergence profile,
    critical radius b, axis ratio q.

    Wei Du, et al. (2020), https://arxiv.org/abs/1507.01819
    """

    param_names = ["b", "a", "a_c","r_c", "center_x", "center_y"]

    def __init__(self):
        super(BPLMajorAxis, self).__init__()
    
    def function(self, x, y, b, a, a_c, r_c,q):
        """Returns the lensing potential.

        :param x: x-coordinate in image plane relative to center (major axis)
        :param y: y-coordinate in image plane relative to center (minor axis)
        :param b: critical radius
        :param t: projected power-law slope
        :param q: axis ratio
        :return: lensing potential
        """

        # 定义矢量化的积分计算函数
        vectorized_quad = np.vectorize(lambda aa, bb: quad(self.integrand_psi, 0, 1, args=(aa, bb, b, a, a_c,  r_c, q))[0])

        # 计算所有参数组合的积分结果
        psi = vectorized_quad(x, y)
        #print(psi)
        return psi

    def derivatives(self, x, y, b, a, a_c, r_c,q):
        """Returns the deflection angles.

        :param x: x-coordinate in image plane relative to center (major axis)
        :param y: y-coordinate in image plane relative to center (minor axis)
        :param b: scale radius
        :param a: slope
        :param a_c: core slope
        :param r_c: break radius
        :param q: axis ratio
        :return: f_x, f_y
        """
        # elliptical radius, eq. (5)
        Z = np.empty(np.shape(x), dtype=complex)
        Z.real = x 
        Z.imag = y
        
        #Z = x*(1+0j)+y*(0+1j)
        #Z = Z+(Z == 0+0j)
      
        R_el = np.sqrt(x**2.*q+y**2./q)
        R_el = R_el + (R_el == 0)
        R_el = np.maximum(R_el, 0.000000001)
        #C = np.maximum(C, 0.000000001)

        #R_el = R / q**0.5
        z = np.sqrt(x**2+y**2)
        zeta2 = (1/q-q)/Z**2
        #z_el = self.zel(R_el,r_c)
        
        #z_el = np.maximum(z_el, 0.000000001)
        C = r_c**2*zeta2
        

        #zeta = np.maximum(zeta, 0.000000001)

        # deflection1, eq. (19)
        #alpha1conj = R_el**2/Z *(b/R_el)**(a-1) * hyp2f1(1/2,(3-a)/2,(5-a)/2,(zeta*R_el)**2)
        
        #phi = np.arctan2(y, q*x)
        #zphi = np.cos(phi)*(1+0j)-np.sin(phi)*(0+1j)
        #t = a - 1
        #alpha1conj = 2.0*np.sqrt(q)*R_el/(1.0+q)*(b/R_el)**t*zphi*self.exhyp2f1(1., 0.5*t, 2.-0.5*t, (q-1.)/(q+1.)*zphi**2.)  # from Tessore & Metcalf 2015
        
        alpha1conj = R_el**2/Z *(b/R_el)**(a-1) * hyp2f1(1/2,(3-a)/2,(5-a)/2,R_el**2*zeta2)
        alpha1 = alpha1conj#.conj()
        # deflection2, eq. (20)
        S0_arr = self.S0(a, a_c, C, R_el, r_c, target_precision=1e-6)
        #S0 = 0
        #coef = rc**2./z*(b/rc)**t*(2.-t)/sf.beta(0.5, 0.5*t)
        alpha2conj = r_c**2/Z *(3-a)/self.Beta_func(a) *(b/r_c)**(a-1) *(2/(3-a_c)*self.F((3-a_c)/2,C) - 2/(3-a)*self.F((3-a)/2,C) - S0_arr)
        alpha2 = alpha2conj#.conj()
        #alpha2 = 0
        #print(2/(3-a_c)*self.F((3-a_c)/2,C),self.F((3-a)/2,C),s0)
        alpha = alpha1 + alpha2

        # return real and imaginary part
        alpha_real = np.nan_to_num(alpha.real, posinf=10**15, neginf=-(10**15))
        alpha_imag = np.nan_to_num(-alpha.imag, posinf=10**15, neginf=-(10**15))

        #print(Z.shape,Z,Z[np.where(Z.imag == np.max(Z.imag)) and np.where(Z.real == np.max(Z.real))], np.where(Z.imag == np.max(Z.imag)))
        
        
        #alpha_real = alpha.real
        #alpha_imag = -alpha.imag
        return alpha_real, alpha_imag

    def hessian(self, x, y, b, a, a_c, r_c,q):
        """Hessian matrix of the lensing potential.

        :param x: x-coordinate in image plane relative to center (major axis)
        :param y: y-coordinate in image plane relative to center (minor axis)
        :param b: scale radius
        :param a: slope
        :param a_c: core slope
        :param r_c: break radius
        :param q: axis ratio
        
        :return: f_xx, f_yy, f_xy
        """
        Z = np.empty(np.shape(x), dtype=complex)
        Z.real = x 
        Z.imag = y
        #R = np.hypot(x*q,y)
        R_el = np.sqrt(x**2.*q+y**2./q)
        #R_el = np.maximum(R_el, 0.000000001)
        z = np.hypot(x,y)
        zeta2 = (1/q-q)/Z**2
        z_el = self.zel(R_el,r_c)
        #z_el = np.maximum(z_el, 0.000000001)
        C = (r_c)**2*zeta2
        
        
        #cos, sin = x / r, y / r
        #cos2, sin2 = cos * cos * 2 - 1, sin * cos * 2

        # convergence, eq. (2)
        kappa1 = (3-a)/2 * (b /R_el)**(a-1)
        
        #if R < r_c:
        #    kappa2 = -(3-a)/self.Beta_func(a) * b /r_c**(a-1) *r *(hyp2f1(a/2,1,3/2,r**2) - hyp2f1(a_c/2,1,3/2,r**2))
        #else:
        #    kappa2 = 0
        kappa2 = self.kappa2func(b, a, a_c, r_c, R_el)
        
        kappa = kappa1 + kappa2
        kappa = np.nan_to_num(kappa, posinf=10**10, neginf=-(10**10))

        # deflection via method
        alpha_x, alpha_y = self.derivatives(x, y, b, a, a_c, r_c, q)
        alpha1 = R_el**2/Z *(b/R_el)**(a-1) * hyp2f1(1/2,(3-a)/2,(5-a)/2,zeta2*(R_el)**2)
        # shear, eq. (17), corrected version from arXiv/corrigendum
        #gamma1_1 = (2 - a) * (alpha_x * cos - alpha_y * sin) / r - kappa1 * cos2
        #gamma2_1 = (2 - a) * (alpha_y * cos + alpha_x * sin) / r - kappa1 * sin2
        gamma1conj = (2-a)*alpha1/Z - kappa1*Z.conj()/Z
        gamma2conj = 2 * (r_c**2/Z**2) * (3-a)/self.Beta_func(a) * (b/r_c)**(a-1) * (
            (2-a_c)/(3-a_c) * self.F((3-a_c)/2,C)
            - (2-a)/(3-a) * self.F((3-a)/2,C) 
            - self.S2(a, a_c, C, R_el, r_c, target_precision=1e-6)
        ) - kappa2*(q*z**2 - (1+q**2)*r_c**2)/(q*Z**2 - (1+q**2)*r_c**2)
        
        gamma = gamma1conj+gamma2conj
        
        gamma_1 = gamma.real
        gamma_2 = -gamma.imag
        
        gamma_1 = np.nan_to_num(gamma_1, posinf=10**10, neginf=-(10**10))
        gamma_2 = np.nan_to_num(gamma_2, posinf=10**10, neginf=-(10**10))

        # second derivatives from convergence and shear
        f_xx = kappa + gamma_1
        f_yy = kappa - gamma_1
        f_xy = gamma_2

        return f_xx, f_xy, f_xy, f_yy
        
    def Beta_func(self, a):
        return beta(1/2,(a-1)/2)
          
    def F(self, a,z):
        if a == 0.5:
            return (spence(1 - np.sqrt(z)) - spence(1 - np.sqrt(-z)))/np.sqrt(z)/2
        else:
            return (1/(1-2*a))*(hyp2f1(a,1,a+1,z) - 2*a*hyp2f1(1/2,1,3/2,z))

    def exhyp2f1(self,a, b, c, z):
        if np.size(z) == 1: z = np.array([z])
        if c-a-b == 0.5:
            zt = np.sqrt(1-z)
            fhyp = (2./(1+zt))**(2.*a)*hyp2f1(2.*a,a-b+0.5,c,(zt-1)/(zt+1))
        else:
            fhyp = hyp2f1(a,b,c,z)
        return fhyp
            
            

    def S0(self,a, a_c, C, R_el, r_c, target_precision):
        if isinstance(R_el, int) or isinstance(R_el, float):
            R_el = np.array([R_el])
        #z_el = self.zel(R_el,r_c)
        result = C*0
        sel = np.where(R_el<r_c)
        #print(R_el.shape,C.shape)
        if len(sel[0]) > 0 and sel[0][0] != 0 :
            #print(sel[0][0],sel)
            #zel2 = z_el[sel]**2
            zel2 = 1. - (R_el[sel]/r_c)**2
            cc = C[sel]
            result[sel] = self.s0arr(a, a_c, zel2, cc,target_precision)
            #result[np.where(R_el>=r_c)] = 0.0 

        return result
    
    
    
    def s0arr(self,alpha, alphac, zel2, c,target_precision):
        # the series term S0 for calculating the deflection angle of BPL model in the core region
        # alpha and alpha_c:the inner and outer density profile slopes for the BPL model
        # zel2: the square of zel=sqrt(1-R_el^2/r_c^2)
        # c:(1-q^2)/q*r_c^2/(x+y*1j)^2
        # zel2 and c should have the same size
        nzc = np.size(zel2)
        if nzc == 1:
            zel2 = np.resize(zel2, 2)
            c = np.resize(c, 2)
        eps = target_precision  # 1e-12
        maxiter = 300
        c1 = c/(c-1.)*zel2
        c2 = np.sqrt(1-c)
        s = 0.0+0.0j
        b = 3./2
        t = 1.0
        tc = 1.0
        a = alpha/2.
        ac = alphac/2.
        nstep = 0
        aks = c*0.0+1.
        while 1:
            aks_tmp = aks*1.
            if nstep < 3:
                h = zel2**b*self.exhyp2f1(0.5, b, b+1, c1)/b
                if nstep == 0:
                    s0 = s*1.
                if nstep == 1:
                    s1 = s*1.
                if nstep == 2:
                    s2 = s*1.
                    aks = s2-(s2-s1)**2./((s2-s1)-(s1-s0))
            else:
                h[sel] = zel2[sel]**b*self.exhyp2f1(0.5, b, b+1, c1[sel])/b
                s0 = s1*1.
                s1 = s2*1.
                s2 = s*1.
                aks[sel] = s2[sel]-(s2[sel]-s1[sel])**2./((s2[sel]-s1[sel])-(s1[sel]-s0[sel]))
                # use the Aitken convergence accelerator
            term = (tc-t)*h
            sel = np.where(abs(aks-aks_tmp) > eps)
            if len(sel[0]) == 0 and nstep > 5:
                break
            if nstep > maxiter:
                break
            s += term
            t = t*a/b
            tc = tc*ac/b
            a += 1
            ac += 1
            b += 1
            h = h*0.
            nstep += 1
        #        print 'abs==,eps=',abs(aks-aks_tmp),eps
        if nzc == 1:
            aks = aks[0]
            c2 = c2[0]
        return aks/c2
    


                       
    def s2arr(self,alpha, alphac, zel2, c,target_precision):
        # the series term S2 for calculating the complex shear of BPL model in the core region
        # alpha and alpha_c:the inner and outer density profile slopes for the BPL model
        # zel2: the square of the zel=sqrt(1-R_el^2/r_c^2)
        # c:(1-q^2)/q*r_c^2/(x+y*1j)^2
        # zel2 and c should have the same size
        nzc = np.size(zel2)
        if nzc == 1:
            zel2 = np.resize(zel2, 2)
            c = np.resize(c, 2)
        eps = target_precision  # 1e-12
        maxiter = 300
        c1 = c/(c-1.)*zel2
        c3 = (1-c)**1.5
        s = 0.0+0.0j
        b = 3./2
        t = 1.0
        tc = 1.0
        a = alpha/2.
        ac = alphac/2.
        nstep = 0
        aks = c*0+1.
        while 1:
            aks_tmp = aks*1.
            if nstep < 3:
                h = zel2**b*self.exhyp2f1(0.5, b, b+1, c1)*(b-0.5)/b
                if nstep == 0:
                    s0 = s*1.
                if nstep == 1:
                    s1 = s*1.
                if nstep == 2:
                    s2 = s*1.
                    aks = s2-(s2-s1)**2./((s2-s1)-(s1-s0))
            else:
                h[sel] = zel2[sel]**b*self.exhyp2f1(0.5, b, b+1, c1[sel])*(b-0.5)/b
                s0 = s1*1.
                s1 = s2*1.
                s2 = s*1.
                aks[sel] = s2[sel]-(s2[sel]-s1[sel])**2./((s2[sel]-s1[sel])-(s1[sel]-s0[sel]))
                # use the Aitken convergence accelerator
            term = (tc-t)*h
            sel = np.where(abs(aks-aks_tmp) > eps)
            if len(sel[0]) == 0 and nstep > 5:
                break
            if nstep > maxiter:
                break
            s += term
            t = t*a/b
            tc = tc*ac/b
            a += 1
            ac += 1
            b += 1
            h = h*0.
            nstep += 1
        #        print 'abs==,eps=',abs(aks-aks_tmp),eps
        if nzc == 1:
            aks = aks[0]
            c3 = c3[0]
        return aks/c3
        
    def S2(self,a, a_c, C, R_el, r_c, target_precision):
        if isinstance(R_el, int) or isinstance(R_el, float):
            R_el = np.array([R_el])
        #z_el = self.zel(R_el,r_c)
        result = C*0
        sel = np.where(R_el<r_c)
        #print(R_el.shape,C.shape)
        if len(sel[0]) > 0 and sel[0][0] != 0 :
            #print(sel[0][0],sel)
            #zel2 = z_el[sel]**2
            zel2 = 1. - (R_el[sel]/r_c)**2
            cc = C[sel]
            result[sel] = self.s2arr(a, a_c, zel2, cc,target_precision)
            #result[np.where(R_el>=r_c)] = 0.0 

        return result

    
    def kappa2func(self, b, a, a_c, r_c, R_el):
        
        if isinstance(R_el, int) or isinstance(R_el, float):
            z_el = self.zel(R_el,r_c)
            if R_el < r_c:
                result = -(3-a)/self.Beta_func(a) * (b /r_c)**(a-1) *z_el *(hyp2f1(a/2,1,3/2,z_el**2) - hyp2f1(a_c/2,1,3/2,z_el**2))
                return result  # 返回计算
            else:
                return 0
        else:
            result = np.empty_like(R_el)
            z_el_array = R_el[R_el < r_c]
            result[R_el < r_c]  = -(3-a)/self.Beta_func(a) * (b /r_c)**(a-1) *z_el_array *(hyp2f1(a/2,1,3/2,z_el_array**2) - hyp2f1(a_c/2,1,3/2,z_el_array**2))
            result[R_el >= r_c] = 0.0
        return result
        
    def zel(self,R_el,r_c):
        if isinstance(R_el, int) or isinstance(R_el, float):
            if R_el < r_c:
                result = (1-R_el**2/r_c**2)**0.5
                return result
            else:
                return 0.0
        else:
            result = np.empty_like(R_el)
            r_el_array = R_el[np.where(R_el < r_c)]
            result[np.where(R_el < r_c)]  = (1-r_el_array**2/r_c**2)**0.5
            result[np.where(R_el >= r_c)] = 0.0
            return result
            
    def kappa_mean(self, R, alpha, alpha_c, b, r_c):
        temp1 = (b / R)**(alpha - 1)
        #kappa0 = -2 / Beta_func(alpha)*(b/r_c)**(alpha - 1)*(r_c/R)**2 * (alpha - alpha_c) / (3 - alpha_c)
        if R < r_c:
            z = (1-(R/r_c)**2)**0.5
            temp3 = 2/3 * (3-alpha)/self.Beta_func(alpha) *(b/r_c)**(alpha - 1) * (r_c/R)**2 * z**3 * (hyp2f1(alpha/2,1,5/2,z**2) - hyp2f1(alpha_c/2,1,5/2,z**2))
            temp4 = -2/3 * (3-alpha)/self.Beta_func(alpha) *(b/r_c)**(alpha - 1) * (r_c/R)**2 * 1**3 * (hyp2f1(alpha/2,1,5/2,1**2) - hyp2f1(alpha_c/2,1,5/2,1**2))
            return temp1 + temp3 + temp4 
        else:
            temp4 = -2/3 * (3-alpha)/self.Beta_func(alpha) *(b/r_c)**(alpha - 1) * (r_c/R)**2 * 1**3 * (hyp2f1(alpha/2,1,5/2,1**2) - hyp2f1(alpha_c/2,1,5/2,1**2))
            return temp1 +temp4 

    def phi_r(self, xi, alpha, alpha_c, b, r_c):
        result = self.kappa_mean(xi, alpha, alpha_c, b, r_c)*xi
        return result
        
    def integrand_psi(self, u, x, y, alpha, alpha_c, b, r_c, q):
        #xi = np.sqrt(u * (x**2 + y**2 / (1 - (1 - q**2) * u)))
        xi = np.sqrt(u * q * (x**2 + y**2 / (1 - (1 - q**2) * u)))
        #return (q / 2) *(xi / u) * self.phi_r(xi, alpha, alpha_c, b, r_c) / np.sqrt(1 - (1 - q**2) * u)
        return (1 / 2) *(xi / u) * self.phi_r(xi, alpha, alpha_c, b, r_c) / np.sqrt(1 - (1 - q**2) * u)


