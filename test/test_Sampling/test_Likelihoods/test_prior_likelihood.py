from lenstronomy.Sampling.Likelihoods.prior_likelihood import PriorLikelihood
import numpy.testing as npt
import numpy as np
from lenstronomy.Util.prob_density import Approx
import pytest


class TestImageLikelihood(object):

    def setup(self):
        self.prior = PriorLikelihood(prior_lens=[[0, 'gamma', 2, 0.1]], prior_source=[], prior_lens_light=[], prior_ps=[],
                                     prior_special=[['source_size', 1, 0.1]])

    def test_logL(self):
        kwargs_lens = [{'gamma': 2}]
        kwargs_cosmo = {'source_size': 1}
        logL = self.prior.logL(kwargs_lens=kwargs_lens, kwargs_source=[], kwargs_lens_light=[], kwargs_ps=[],
                               kwargs_special=kwargs_cosmo)
        assert logL == 0

        kwargs_lens = [{'gamma': 2.1}]
        kwargs_cosmo = {'source_size': 1.1}
        logL = self.prior.logL(kwargs_lens=kwargs_lens, kwargs_source=[], kwargs_lens_light=[], kwargs_ps=[],
                               kwargs_special=kwargs_cosmo)
        npt.assert_almost_equal(logL, -1, decimal=8)

    def gauss(self, x, mean, simga):
        return np.exp(-((x-mean)/(simga))**2/2) / np.sqrt(2*np.pi) / simga

    def test_kde_prior(self):
        x_array = np.linspace(1., 3., 200)
        sigma = .2
        mean = 2
        pdf_array = self.gauss(x_array, mean=mean, simga=sigma)
        approx = Approx(x_array, pdf_array)
        sample = approx.draw(n=50000)
        prior = PriorLikelihood(prior_lens_kde=[[0, 'gamma', sample]])

        kwargs_lens = [{'gamma': 2}]
        logL = prior.logL(kwargs_lens=kwargs_lens, kwargs_source=[], kwargs_lens_light=[], kwargs_ps=[])

        kwargs_lens = [{'gamma': 2.2}]
        logL_sigma = prior.logL(kwargs_lens=kwargs_lens, kwargs_source=[], kwargs_lens_light=[], kwargs_ps=[])
        delta_log = logL - logL_sigma
        npt.assert_almost_equal(delta_log, 0.5, decimal=1)

        kwargs_lens = [{'gamma': 2.4}]
        logL_sigma = prior.logL(kwargs_lens=kwargs_lens, kwargs_source=[], kwargs_lens_light=[], kwargs_ps=[])
        delta_log = logL - logL_sigma
        npt.assert_almost_equal(delta_log, 2, decimal=1)


if __name__ == '__main__':
    pytest.main()
