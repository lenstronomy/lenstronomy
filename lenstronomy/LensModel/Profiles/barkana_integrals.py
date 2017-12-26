__author__ = 'sibirrer'

import numpy as np
import pickle
import os.path
from scipy import integrate

import lenstronomy.Util.util as util


class BarkanaIntegrals(object):

    def I1(self, nu1, nu2, s_, gamma):
        """
        integral of Barkana et al. (18)
        :param nu2:
        :param s_:
        :param gamma:
        :return:
        """
        return self.I1_numeric(nu1, nu2, s_, gamma)
        # if not hasattr(self,'I1_interp'):
        #     self.open_I1()
        # return self.I1_interp(nu2, s_, gamma)

    def write_I1(self):
        self.interp_I1()  # creating self.I1_interp
        f = open('Interpolations/I1_interp.txt', 'wb')
        pickle.dump(self.I1_interp,f)
        f.close()
        print('file I1_interp.txt new writen')

    def open_I1(self):
        if not os.path.isfile('Interpolations/I1_interp.txt'):
            self.write_I1()
        f = open('Interpolations/I1_interp.txt','rb')
        self.I1_interp = pickle.load(f)
        f.close()
        print('I1 opened')

    def interp_I1(self):
        pass

    def _I1_intg(self, nu, s_, gamma):
        return nu**(-gamma)* self._f(nu-s_)

    def I1_numeric(self, nu1, nu2, s_, gamma):
        nu1 = util.mk_array(nu1)
        nu2 = util.mk_array(nu2)
        s_ = util.mk_array(s_)
        I1 = np.empty_like(nu2)
        for i in range(len(nu2)):
            nu_min = nu1[i]
            nu_max = nu2[i]
            result, error = integrate.quad(self._I1_intg, nu_min, nu_max, args=(s_[i], gamma))
            I1[i] = result
        return I1

    def I2(self, nu1, nu2, s_, gamma):
        """
        integral of Barkana et al. (18)
        :param nu2:
        :param s_:
        :param gamma:
        :return:
        """
        return self.I2_numeric(nu1, nu2, s_, gamma)
        # if not hasattr(self,'I2_interp'):
        #     self.open_I2()
        # return self.I2_interp(nu2, s_, gamma)

    def write_I2(self):
        self.interp_I2()  # creating self.I2_interp
        f = open('Interpolations/I2_interp.txt', 'wb')
        pickle.dump(self.I2_interp,f)
        f.close()
        print('file I2_interp.txt new writen')

    def open_I2(self):
        if not os.path.isfile('Interpolations/I2_interp.txt'):
            self.write_I2()
        f = open('Interpolations/I2_interp.txt','rb')
        self.I2_interp = pickle.load(f)
        f.close()
        print('I1 opened')

    def interp_I2(self):
        pass

    def _I2_intg(self, nu, s_, gamma):
        return nu**(-gamma)* self._f(s_-nu)

    def I2_numeric(self, nu1, nu2, s_, gamma):
        nu1 = util.mk_array(nu1)
        nu2 = util.mk_array(nu2)
        s_ = util.mk_array(s_)
        I2 = np.empty_like(nu2)
        for i in range(len(nu2)):
            nu_min = nu1[i]
            nu_max = nu2[i]
            result, error = integrate.quad(self._I2_intg,nu_min,nu_max,args=(s_[i], gamma))
            I2[i] = result
        return I2


    def I3(self, nu2, s_, gamma):
        """
        integral of Barkana et al. (23)
        :param nu2:
        :param s_:
        :param gamma:
        :return:
        """
        return self.I3_numeric(nu2, s_, gamma)
        # if not hasattr(self,'I3_interp'):
        #     self.open_I3()
        # return self.I3_interp(nu2, s_, gamma)

    def write_I3(self):
        self.interp_I3()  # creating self.I3_interp
        f = open('Interpolations/I3_interp.txt', 'wb')
        pickle.dump(self.I3_interp,f)
        f.close()
        print('file I3_interp.txt new writen')

    def open_I3(self):
        if not os.path.isfile('Interpolations/I3_interp.txt'):
            self.write_I3()
        f = open('Interpolations/I3_interp.txt','rb')
        self.I3_interp = pickle.load(f)
        f.close()
        print('I3 opened')

    def interp_I3(self):
        pass

    def _I3_intg(self, nu, s_, gamma):
        return nu**(-gamma) * self._f_deriv(nu-s_)

    def I3_numeric(self, nu2, s_, gamma):
        nu_min = 0
        nu2 = util.mk_array(nu2)
        s_ = util.mk_array(s_)
        I3 = np.empty_like(nu2)
        for i in range(len(nu2)):
            nu_max = nu2[i]
            result, error = integrate.quad(self._I3_intg,nu_min,nu_max,args=(s_[i], gamma))
            I3[i] = result
        return I3



    def I4(self, nu2, s_, gamma):
        """
        integral of Barkana et al. (23)
        :param nu2:
        :param s_:
        :param gamma:
        :return:
        """
        return self.I4_numeric(nu2, s_, gamma)
        # if not hasattr(self,'I4_interp'):
        #     self.open_I4()
        # return self.I4_interp(nu2, s_, gamma)

    def write_I4(self):
        self.interp_I4()  # creating self.I4_interp
        f = open('Interpolations/I4_interp.txt', 'wb')
        pickle.dump(self.I4_interp,f)
        f.close()
        print('file I4_interp.txt new writen')

    def open_I4(self):
        if not os.path.isfile('Interpolations/I4_interp.txt'):
            self.write_I4()
        f = open('Interpolations/I4_interp.txt','rb')
        self.I4_interp = pickle.load(f)
        f.close()
        print('I4 opened')

    def interp_I4(self):
        pass

    def _I4_intg(self, nu, s_, gamma):
        return nu**(-gamma) * self._f_deriv(s_-nu)

    def I4_numeric(self, nu2, s_, gamma):
        nu_min = 0
        nu2 = util.mk_array(nu2)
        s_ = util.mk_array(s_)
        I4 = np.empty_like(nu2)
        for i in range(len(nu2)):
            nu_max = nu2[i]
            result, error = integrate.quad(self._I4_intg,nu_min,nu_max,args=(s_[i], gamma))
            I4[i] = result
        return I4

    def _f(self, mu):
        """
        f(mu) function (eq 15 in Barkana et al.)
        :param mu:
        :return:
        """
        return np.sqrt(1/np.sqrt(1+mu**2) - mu/(mu**2+1))

    def _f_deriv(self, mu):
        """
        f'(mu) function (derivative of eq 15 in barkana et al.)
        :param mu:
        :return:
        """
        a = np.sqrt(mu**2+1)
        term1 = -mu*np.sqrt(a-mu) / a**3
        term2 = -(a -mu) / (2*(mu**2+1)*np.sqrt(a-mu))
        return term1 + term2