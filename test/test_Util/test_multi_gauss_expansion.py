__author__ = 'sibirrer'

import lenstronomy.Util.multi_gauss_expansion as mge

import numpy as np
import numpy.testing as npt
from lenstronomy.LightModel.Profiles.sersic import Sersic
from lenstronomy.LightModel.Profiles.hernquist import Hernquist
from lenstronomy.LightModel.Profiles.gaussian import MultiGaussian
import pytest

class TestMGE(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.sersic = Sersic()
        self.multiGaussian = MultiGaussian()

    def test_mge_1d_sersic(self):
        n_comp = 30
        r_sersic = 1.
        n_sersic = 3.7
        I0_sersic = 1.
        rs = np.logspace(-2., 1., 50) * r_sersic
        ss = self.sersic.function(rs, np.zeros_like(rs), I0_sersic=I0_sersic, n_sersic=n_sersic, R_sersic=r_sersic)

        amplitudes, sigmas, norm = mge.mge_1d(rs, ss, N=n_comp)
        ss_mge = self.multiGaussian.function(rs, np.zeros_like(rs), amp=amplitudes, sigma=sigmas)
        #print((ss - ss_mge)/ss)
        for i in range(10, len(ss)-10):
            #print(rs[i])
            npt.assert_almost_equal((ss_mge[i]-ss[i])/ss[i], 0, decimal=1)

    def test_mge_sersic_radius(self):
        n_comp = 30
        r_sersic = .5
        n_sersic = 3.7
        I0_sersic = 1.
        rs = np.logspace(-2., 1., 50) * r_sersic
        ss = self.sersic.function(rs, np.zeros_like(rs), I0_sersic=I0_sersic, n_sersic=n_sersic, R_sersic=r_sersic)

        amplitudes, sigmas, norm = mge.mge_1d(rs, ss, N=n_comp)
        ss_mge = self.multiGaussian.function(rs, np.zeros_like(rs), amp=amplitudes, sigma=sigmas)
        print((ss - ss_mge)/(ss+ ss_mge))
        for i in range(10, len(ss)-10):
            #print(rs[i])
            npt.assert_almost_equal((ss_mge[i]-ss[i])/(ss[i]), 0, decimal=1)

    def test_mge_sersic_n_sersic(self):
        n_comp = 20
        r_sersic = 1.5
        n_sersic = .5
        I0_sersic = 1.
        rs = np.logspace(-2., 1., 50) * r_sersic
        ss = self.sersic.function(rs, np.zeros_like(rs), I0_sersic=I0_sersic, n_sersic=n_sersic, R_sersic=r_sersic)

        amplitudes, sigmas, norm = mge.mge_1d(rs, ss, N=n_comp)
        ss_mge = self.multiGaussian.function(rs, np.zeros_like(rs), amp=amplitudes, sigma=sigmas)
        for i in range(10, len(ss)-10):
            npt.assert_almost_equal((ss_mge[i]-ss[i])/(ss[i]+ss_mge[i]), 0, decimal=1)

        n_comp = 20
        r_sersic = 1.5
        n_sersic =3.5
        I0_sersic = 1.
        rs = np.logspace(-2., 1., 50) * r_sersic
        ss = self.sersic.function(rs, np.zeros_like(rs), I0_sersic=I0_sersic, n_sersic=n_sersic, R_sersic=r_sersic)

        amplitudes, sigmas, norm = mge.mge_1d(rs, ss, N=n_comp)
        ss_mge = self.multiGaussian.function(rs, np.zeros_like(rs), amp=amplitudes, sigma=sigmas)
        for i in range(10, len(ss)-10):
            npt.assert_almost_equal((ss_mge[i]-ss[i])/(ss[i]+ss_mge[i]), 0, decimal=1)

    def test_hernquist(self):
        hernquist = Hernquist()
        n_comp = 20
        sigma0 = 1
        r_eff = 1.5
        rs = np.logspace(-2., 1., 50) * r_eff * 0.5
        ss = hernquist.function(rs, np.zeros_like(rs), sigma0, Rs=r_eff)
        amplitudes, sigmas, norm = mge.mge_1d(rs, ss, N=n_comp)
        ss_mge = self.multiGaussian.function(rs, np.zeros_like(rs), amp=amplitudes, sigma=sigmas)
        for i in range(10, len(ss)-10):
            npt.assert_almost_equal((ss_mge[i]-ss[i])/(ss[i]+ss_mge[i]), 0, decimal=2)

    def test_hernquist_deprojection(self):
        hernquist = Hernquist()
        n_comp = 20
        sigma0 = 1
        r_eff = 1.5
        rs = np.logspace(-2., 1., 50) * r_eff * 0.5
        ss = hernquist.function(rs, np.zeros_like(rs), sigma0, Rs=r_eff)
        amplitudes, sigmas, norm = mge.mge_1d(rs, ss, N=n_comp)
        amplitudes_3d, sigmas_3d = mge.de_projection_3d(amplitudes, sigmas)
        ss_3d_mge = self.multiGaussian.function(rs, np.zeros_like(rs), amp=amplitudes_3d, sigma=sigmas_3d)
        ss_3d_mulit = self.multiGaussian.light_3d(rs, amp=amplitudes, sigma=sigmas)
        for i in range(10, len(ss_3d_mge)):
            npt.assert_almost_equal((ss_3d_mge[i] - ss_3d_mulit[i]) / (ss_3d_mulit[i] + ss_3d_mge[i]), 0, decimal=1)

        ss_3d = hernquist.light_3d(rs, sigma0, Rs=r_eff)
        for i in range(10, len(ss_3d) -10):
            npt.assert_almost_equal((ss_3d_mge[i] - ss_3d[i]) / (ss_3d[i] + ss_3d_mge[i]), 0, decimal=1)

    def test_spemd(self):
        from lenstronomy.LensModel.Profiles.spep import SPEP
        from lenstronomy.LensModel.Profiles.multi_gaussian_kappa import MultiGaussian_kappa
        spep = SPEP()
        mge_kappa = MultiGaussian_kappa()
        n_comp = 8
        theta_E = 1.41
        kwargs = {'theta_E': theta_E, 'q': 1.,
           'phi_G': 0.99, 'gamma': 1.61}
        rs = np.logspace(-2., 1., 100) * theta_E
        f_xx, f_yy, f_xy = spep.hessian(rs, 0, **kwargs)
        kappa = 1/2. * (f_xx + f_yy)
        amplitudes, sigmas, norm = mge.mge_1d(rs, kappa, N=n_comp)
        kappa_mge = self.multiGaussian.function(rs, np.zeros_like(rs), amp=amplitudes, sigma=sigmas)
        f_xx_mge, f_yy_mge, f_xy_mge = mge_kappa.hessian(rs, np.zeros_like(rs), amp=amplitudes, sigma=sigmas)
        for i in range(0, 80):
            npt.assert_almost_equal(kappa_mge[i], 1./2 * (f_xx_mge[i] + f_yy_mge[i]), decimal=1)
            npt.assert_almost_equal((kappa[i] - kappa_mge[i])/kappa[i], 0, decimal=1)

        f_ = spep.function(theta_E, 0, **kwargs)
        f_mge = mge_kappa.function(theta_E, 0, sigma=sigmas, amp=amplitudes)
        npt.assert_almost_equal(f_mge/f_, 1, decimal=2)

    def test_example(self):
        n_comp = 10
        rs = np.array([0.01589126,   0.01703967,   0.01827108,   0.01959148,
         0.0210073 ,   0.02252544,   0.02415329,   0.02589879,
         0.02777042,   0.02977731,   0.03192923,   0.03423667,
         0.03671086,   0.03936385,   0.04220857,   0.04525886,
         0.0485296 ,   0.0520367 ,   0.05579724,   0.05982956,
         0.06415327,   0.06878945,   0.07376067,   0.07909115,
         0.08480685,   0.09093561,   0.09750727,   0.10455385,
         0.11210966,   0.12021152,   0.12889887,   0.13821403,
         0.14820238,   0.15891255,   0.17039672,   0.18271082,
         0.19591482,   0.21007304,   0.22525444,   0.24153295,
         0.25898787,   0.2777042 ,   0.29777311,   0.31929235,
         0.34236672,   0.36710861,   0.39363853,   0.42208569,
         0.45258865,   0.48529597,   0.52036697,   0.55797244,
         0.59829556,   0.64153272,   0.6878945 ,   0.73760673,
         0.79091152,   0.8480685 ,   0.90935605,   0.97507269,
         1.04553848,   1.12109664,   1.20211518,   1.28898871,
         1.38214034,   1.48202378,   1.58912553,   1.70396721,
         1.82710819,   1.95914822,   2.10073042,   2.25254437,
         2.4153295 ,   2.58987865,   2.77704199,   2.9777311 ,
         3.19292345,   3.42366716,   3.67108607,   3.93638527,
         4.22085689,   4.5258865 ,   4.85295974,   5.20366966,
         5.57972441,   5.98295559,   6.41532717,   6.87894505,
         7.37606729,   7.90911519,   8.48068497,   9.09356051,
         9.75072687,  10.45538481,  11.21096643,  12.02115183,
        12.88988708,  13.82140341,  14.82023784,  15.89125526])
        kappa = np.array([ 12.13776067,  11.60484966,  11.09533396,  10.60818686,
        10.14242668,   9.69711473,   9.27135349,   8.86428482,
         8.47508818,   8.10297905,   7.7472073 ,   7.40705574,
         7.08183863,   6.77090034,   6.47361399,   6.18938022,
         5.917626  ,   5.65780342,   5.40938864,   5.1718808 ,
         4.94480104,   4.72769151,   4.52011448,   4.3216514 ,
         4.13190214,   3.9504841 ,   3.77703149,   3.61119459,
         3.45263901,   3.30104507,   3.1561071 ,   3.01753287,
         2.88504297,   2.75837025,   2.63725931,   2.52146595,
         2.41075668,   2.30490829,   2.20370736,   2.10694982,
         2.01444058,   1.92599312,   1.84142909,   1.76057799,
         1.6832768 ,   1.60936965,   1.53870751,   1.47114792,
         1.40655465,   1.34479745,   1.28575181,   1.22929867,
         1.17532421,   1.12371958,   1.07438074,   1.02720821,
         0.98210687,   0.93898578,   0.897758  ,   0.85834039,
         0.82065349,   0.78462129,   0.75017114,   0.71723359,
         0.68574222,   0.65563353,   0.62684681,   0.59932403,
         0.57300967,   0.5478507 ,   0.52379638,   0.5007982 ,
         0.47880979,   0.45778683,   0.43768691,   0.41846951,
         0.40009589,   0.38252899,   0.3657334 ,   0.34967525,
         0.33432216,   0.31964317,   0.30560868,   0.29219041,
         0.27936129,   0.26709545,   0.25536817,   0.24415579,
         0.23343571,   0.22318631,   0.21338694,   0.20401782,
         0.19506006,   0.18649562,   0.17830721,   0.17047832,
         0.16299318,   0.15583668,   0.14899441,   0.14245255])
        amplitudes, sigmas, norm = mge.mge_1d(rs, kappa, N=n_comp)

    def test_nfw_sersic(self):
        kwargs_lens_nfw = {'theta_Rs': 1.4129647849966354, 'Rs': 7.0991113634274736}
        kwargs_lens_sersic = {'k_eff': 0.24100561407593576, 'n_sersic': 1.8058507329346063, 'r_eff': 1.0371803141813705}
        from lenstronomy.LensModel.Profiles.nfw import NFW
        from lenstronomy.LensModel.Profiles.sersic import Sersic
        nfw = NFW()
        sersic = Sersic()
        theta_E = 1.5
        n_comp = 10
        rs = np.logspace(-2., 1., 100) * theta_E
        f_xx_nfw, f_yy_nfw, f_xy_nfw = nfw.hessian(rs, 0, **kwargs_lens_nfw)
        f_xx_s, f_yy_s, f_xy_s = sersic.hessian(rs, 0, **kwargs_lens_sersic)
        kappa = 1 / 2. * (f_xx_nfw + f_xx_s + f_yy_nfw + f_yy_s)
        amplitudes, sigmas, norm = mge.mge_1d(rs, kappa, N=n_comp)
        kappa_mge = self.multiGaussian.function(rs, np.zeros_like(rs), amp=amplitudes, sigma=sigmas)
        from lenstronomy.LensModel.Profiles.multi_gaussian_kappa import MultiGaussian_kappa
        mge_kappa = MultiGaussian_kappa()
        f_xx_mge, f_yy_mge, f_xy_mge = mge_kappa.hessian(rs, np.zeros_like(rs), amp=amplitudes, sigma=sigmas)
        for i in range(0, 80):
            npt.assert_almost_equal(kappa_mge[i], 1. / 2 * (f_xx_mge[i] + f_yy_mge[i]), decimal=1)
            npt.assert_almost_equal((kappa[i] - kappa_mge[i]) / kappa[i], 0, decimal=1)

        f_nfw = nfw.function(theta_E, 0, **kwargs_lens_nfw)
        f_s = sersic.function(theta_E, 0, **kwargs_lens_sersic)
        f_mge = mge_kappa.function(theta_E, 0, sigma=sigmas, amp=amplitudes)
        npt.assert_almost_equal(f_mge / (f_nfw + f_s), 1, decimal=2)


if __name__ == '__main__':
    pytest.main()