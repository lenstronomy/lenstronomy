import numpy as np
import numpy.testing as npt
import pytest
# import lenstronomy module dealing with cosmological quantities in a lens system
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
# import KDELikelihood module
from lenstronomy.Cosmo.kde_likelihood import KDELikelihood
# import astropy.cosmology class
from astropy.cosmology import FlatLambdaCDM


class TestKDELikelihood(object):

    def setup(self):
        # set up seed
        np.random.seed(seed=41)
        pass

    def test_kde_likelihood(self):
        # define redshift of lens and source
        z_L = 0.8
        z_S = 3.0

        # define the "truth"
        H0_true = 70
        omega_m_true = 0.3
        # setup the true cosmology
        cosmo = FlatLambdaCDM(H0_true, Om0=omega_m_true)
        lensCosmo = LensCosmo(z_L, z_S, cosmo=cosmo)
        # compute the true angular diameter distances
        Dd_true = lensCosmo.D_d
        D_dt_true = lensCosmo.D_dt

        # define a measurement uncertainty/spread in the posteriors
        # the example contains uncorrelated Gaussians
        sigma_Dd = 100
        sigma_Ddt = 100

        # make a realization of the posterior sample, centered around the "truth" with given uncertainties
        num_samples = 50000
        D_dt_samples = np.random.normal(D_dt_true, sigma_Ddt, num_samples)
        D_d_samples = np.random.normal(Dd_true, sigma_Dd, num_samples)

        # initialize a KDELikelihood class with the posterior sample
        kdeLikelihood = KDELikelihood(D_d_samples, D_dt_samples, kde_type='scipy_gaussian',  bandwidth=2)
        # evaluate the maximum likelihood (arbitrary normalization!)
        logL_max = kdeLikelihood.logLikelihood(Dd_true, D_dt_true)
        # evaluate the likelihood 1-sigma away from Dd
        logL_sigma = kdeLikelihood.logLikelihood(Dd_true + sigma_Dd, D_dt_true)
        # compute likelihood ratio
        delta_log = logL_max - logL_sigma
        # check whether likelihood ratio is consistent with input distribution
        # (in relative percent level in the likelihoods)
        npt.assert_almost_equal(delta_log, 0.5, decimal=2)

        # test the same in D_dt dimension
        logL_sigma = kdeLikelihood.logLikelihood(Dd_true, D_dt_true + sigma_Ddt)
        # compute likelihood ratio
        delta_log = logL_max - logL_sigma
        # check whether likelihood ratio is consistent with input distribution
        npt.assert_almost_equal(delta_log, 0.5, decimal=2)


if __name__ == '__main__':
    pytest.main()


import corner.corner