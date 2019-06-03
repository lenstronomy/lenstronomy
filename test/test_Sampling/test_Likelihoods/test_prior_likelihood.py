from lenstronomy.Sampling.Likelihoods.prior_likelihood import PriorLikelihood
import numpy.testing as npt
import pytest


class TestImageLikelihood(object):

    def setup(self):
        self.prior = PriorLikelihood(prior_lens=[[0, 'gamma', 2, 0.1]], prior_source=[], prior_lens_light=[], prior_ps=[],
                                     prior_cosmo=[['source_size', 1, 0.1]])

    def test_logL(self):
        kwargs_lens = [{'gamma': 2}]
        kwargs_cosmo = {'source_size': 1}
        logL = self.prior.logL(kwargs_lens=kwargs_lens, kwargs_source=[], kwargs_lens_light=[], kwargs_ps=[],
                        kwargs_cosmo=kwargs_cosmo)
        assert logL == 0

        kwargs_lens = [{'gamma': 2.1}]
        kwargs_cosmo = {'source_size': 1.1}
        logL = self.prior.logL(kwargs_lens=kwargs_lens, kwargs_source=[], kwargs_lens_light=[], kwargs_ps=[],
                               kwargs_cosmo=kwargs_cosmo)
        npt.assert_almost_equal(logL, -1, decimal=8)


if __name__ == '__main__':
    pytest.main()
