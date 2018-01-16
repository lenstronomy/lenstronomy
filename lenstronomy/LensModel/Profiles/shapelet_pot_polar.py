__author__ = 'sibirrer'

# description of the polar shapelets in potential space

import numpy as np
import scipy.special
import math

import lenstronomy.Util.param_util as param_util


class PolarShapelets(object):
    """
    this class contains the function and the derivatives of the Singular Isothermal Sphere
    """
    def __init__(self):
        n = 10
        self.poly = [[[] for i in range(n)] for i in range(n)]
        for i in range(0,n):
            for j in range(0,n):
                self.poly[i][j] = scipy.special.genlaguerre(i, j)

    def function(self, x, y, coeffs, beta, center_x=0, center_y=0):
        shapelets = self._createShapelet(coeffs)
        r, phi = param_util.cart2polar(x, y, center=np.array([center_x, center_y]))
        f_ = self._shapeletOutput(r, phi, beta, shapelets)
        return f_

    def derivatives(self, x, y, coeffs, beta, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function
        """
        shapelets = self._createShapelet(coeffs)
        r, phi = param_util.cart2polar(x, y, center=np.array([center_x, center_y]))
        alpha1_shapelets, alpha2_shapelets = self._alphaShapelets(shapelets, beta)
        f_x = self._shapeletOutput(r, phi, beta, alpha1_shapelets)
        f_y = self._shapeletOutput(r, phi, beta, alpha2_shapelets)
        return f_x, f_y

    def hessian(self, x, y, coeffs, beta, center_x=0, center_y=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        shapelets = self._createShapelet(coeffs)
        r, phi = param_util.cart2polar(x, y, center=np.array([center_x, center_y]))
        kappa_shapelets=self._kappaShapelets(shapelets, beta)
        gamma1_shapelets, gamma2_shapelets=self._gammaShapelets(shapelets, beta)
        kappa_value=self._shapeletOutput(r, phi, beta, kappa_shapelets)
        gamma1_value=self._shapeletOutput(r, phi, beta, gamma1_shapelets)
        gamma2_value=self._shapeletOutput(r, phi, beta, gamma2_shapelets)
        f_xx = kappa_value + gamma1_value
        f_xy = gamma2_value
        f_yy = kappa_value - gamma1_value
        return f_xx, f_yy, f_xy

    def _createShapelet(self,coeff):
        """
        returns a shapelet array out of the coefficients *a, up to order l

        :param num_l: order of shapelets
        :type num_l: int.
        :param coeff: shapelet coefficients
        :type coeff: floats
        :returns:  complex array
        :raises: AttributeError, KeyError
        """
        n_coeffs = len(coeff)
        num_l = self._get_num_l(n_coeffs)
        shapelets=np.zeros((num_l+1,num_l+1),'complex')
        nl=0
        k=0
        i=0
        while i < len(coeff):
            if i%2==0:
                shapelets[nl][k]+=coeff[i]/2.
                shapelets[k][nl]+=coeff[i]/2.
                if k==nl:
                    nl+=1
                    k=0
                    i+=1
                    continue
                else:
                    k+=1
                    i+=1
                    continue
            else:
                shapelets[nl][k] += 1j*coeff[i]/2.
                shapelets[k][nl] -= 1j*coeff[i]/2.
                i+=1
        return shapelets

    def _shapeletOutput(self, r, phi, beta, shapelets):
        """
        returns the the numerical values of a set of shapelets at polar coordinates
        :param shapelets: set of shapelets [l=,r=,a_lr=]
        :type shapelets: array of size (n,3)
        :param coordPolar: set of coordinates in polar units
        :type coordPolar: array of size (n,2)
        :returns:  array of same size with coords [r,phi]
        :raises: AttributeError, KeyError
        """
        if type(r) == float or type(r) == int or type(r) == type(np.float64(1)) or len(r) <= 1:
            values = 0.
        else:
            values = np.zeros(len(r), 'complex')
        for nl in range(0,len(shapelets)): #sum over different shapelets
            for nr in range(0,len(shapelets)):
                value = shapelets[nl][nr]*self._chi_lr(r, phi, nl, nr, beta)
                values += value
        return values.real

    def _chi_lr(self,r, phi, nl,nr,beta):
        """
        computes the generalized polar basis function in the convention of Massey&Refregier eqn 8

        :param nl: left basis
        :type nl: int
        :param nr: right basis
        :type nr: int
        :param beta: beta --the characteristic scale typically choosen to be close to the size of the object.
        :type beta: float.
        :param coord: coordinates [r,phi]
        :type coord: array(n,2)
        :returns:  values at positions of coordinates.
        :raises: AttributeError, KeyError
        """
        m=int((nr-nl).real)
        n=int((nr+nl).real)
        p=int((n-abs(m))/2)
        p2=int((n+abs(m))/2)
        q=int(abs(m))
        if p % 2==0: #if p is even
            prefac=1
        else:
            prefac=-1
        prefactor=prefac/beta**(abs(m)+1)*np.sqrt(math.factorial(p)/(np.pi*math.factorial(p2)))
        poly=self.poly[p][q]
        return prefactor*r**q*poly((r/beta)**2)*np.exp(-(r/beta)**2/2)*np.exp(-1j*m*phi)

    def _kappaShapelets(self, shapelets, beta):
        """
        calculates the convergence kappa given lensing potential shapelet coefficients (laplacian/2)
        :param shapelets: set of shapelets [l=,r=,a_lr=]
        :type shapelets: array of size (n,3)
        :returns:  set of kappa shapelets.
        :raises: AttributeError, KeyError
        """
        output=np.zeros((len(shapelets)+1,len(shapelets)+1),'complex')
        for nl in range(0,len(shapelets)):
            for nr in range(0,len(shapelets)):
                a_lr=shapelets[nl][nr]
                if nl>0:
                    output[nl-1][nr+1]+=a_lr*np.sqrt(nl*(nr+1))/2
                    if nr>0:
                        output[nl-1][nr-1]+=a_lr*np.sqrt(nl*nr)/2
                output[nl+1][nr+1]+=a_lr*np.sqrt((nl+1)*(nr+1))/2
                if nr>0:
                    output[nl+1][nr-1]+=a_lr*np.sqrt((nl+1)*nr)/2
        return output/beta**2


    def _alphaShapelets(self,shapelets, beta):
        """
        calculates the deflection angles given lensing potential shapelet coefficients (laplacian/2)
        :param shapelets: set of shapelets [l=,r=,a_lr=]
        :type shapelets: array of size (n,3)
        :returns:  set of alpha shapelets.
        :raises: AttributeError, KeyError
        """
        output_x = np.zeros((len(shapelets)+1, len(shapelets)+1), 'complex')
        output_y = np.zeros((len(shapelets)+1, len(shapelets)+1), 'complex')
        for nl in range(0,len(shapelets)):
            for nr in range(0,len(shapelets)):
                a_lr=shapelets[nl][nr]
                output_x[nl][nr+1]-=a_lr*np.sqrt(nr+1)/2
                output_y[nl][nr+1]-=a_lr*np.sqrt(nr+1)/2*1j
                output_x[nl+1][nr]-=a_lr*np.sqrt(nl+1)/2
                output_y[nl+1][nr]+=a_lr*np.sqrt(nl+1)/2*1j
                if nl>0:
                    output_x[nl-1][nr]+=a_lr*np.sqrt(nl)/2
                    output_y[nl-1][nr]-=a_lr*np.sqrt(nl)/2*1j
                if nr>0:
                    output_x[nl][nr-1]+=a_lr*np.sqrt(nr)/2
                    output_y[nl][nr-1]+=a_lr*np.sqrt(nr)/2*1j
        return output_x/beta,output_y/beta  #attention complex numbers!!!!

    def _gammaShapelets(self,shapelets, beta):
        """
        calculates the shear gamma given lensing potential shapelet coefficients
        :param shapelets: set of shapelets [l=,r=,a_lr=]
        :type shapelets: array of size (n,3)
        :returns:  set of alpha shapelets.
        :raises: AttributeError, KeyError
        """
        output_x = np.zeros((len(shapelets)+2,len(shapelets)+2),'complex')
        output_y = np.zeros((len(shapelets)+2,len(shapelets)+2),'complex')
        for nl in range(0, len(shapelets)):
            for nr in range(0, len(shapelets)):
                a_lr = shapelets[nl][nr]
                output_x[nl+2][nr] += a_lr*np.sqrt((nl+1)*(nl+2))/2
                output_x[nl][nr+2] += a_lr*np.sqrt((nr+1)*(nr+2))/2
                output_x[nl][nr] += a_lr*(1-(nr+1)-(nl+1))
                if nl>1:
                    output_x[nl-2][nr] += a_lr*np.sqrt((nl)*(nl-1))/2
                if nr>1:
                    output_x[nl][nr-2] += a_lr*np.sqrt((nr)*(nr-1))/2

                output_y[nl+2][nr] += a_lr*np.sqrt((nl+1)*(nl+2))*1j/4
                output_y[nl][nr+2] -= a_lr*np.sqrt((nr+1)*(nr+2))*1j/4
                if nl>0:
                    output_y[nl-1][nr+1] += a_lr*np.sqrt((nl)*(nr+1))*1j/2
                if nr>0:
                    output_y[nl+1][nr-1] -= a_lr*np.sqrt((nr)*(nl+1))*1j/2
                if nl>1:
                    output_y[nl-2][nr] -= a_lr*np.sqrt((nl)*(nl-1))*1j/4
                if nr>1:
                    output_y[nl][nr-2] += a_lr*np.sqrt((nr)*(nr-1))*1j/4
        return output_x/beta**2, output_y/beta**2  #attention complex numbers!!!!

    def _get_num_l(self, n_coeffs):
        """

        :param n_coeffs: number of coeffs
        :return: number of n_l of order of the shapelets
        """
        num_l = int(round((math.sqrt(8*n_coeffs + 9)-3)/2 +0.499))
        return num_l
