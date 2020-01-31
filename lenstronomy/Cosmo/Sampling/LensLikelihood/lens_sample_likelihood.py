
from lenstronomy.Cosmo.Sampling.LensLikelihood.lens_likelihood import LensLikelihood


class LensSampleLikelihood(object):
    """
    class to evaluate the likelihood of a cosmology given a sample of angular diameter posteriors
    Currently this class does not include possible covariances between the lens samples
    """
    def __init__(self, kwargs_lens_list):
        """

        :param kwargs_lens_list: keyword argument list specifying the arguments of the LensLikelihood class
        """
        self._lens_list = []
        for kwargs_lens in kwargs_lens_list:
            self._lens_list.append(LensLikelihood(**kwargs_lens))

    def log_likelihood(self, cosmo, gamma_ppn=1, lambda_mst=1, kappa_ext=0, aniso_param=None):
        """

        :param cosmo: astropy.cosmology instance
        :param gamma_ppn: post-newtonian gravity parameter (=1 is GR)
        :param lambda_mst: overall global mass-sheet transform applied on the sample,
        lambda_mst=1 corresponds to the input model
        :param aniso_param: global stellar anisotropy parameter
        :param kappa_ext: external convergence to be added on top of the D_dt posterior
        :return: log likelihood of the combined lenses
        """
        logL = 0
        for lens in self._lens_list:
            logL += lens.lens_log_likelihood(cosmo=cosmo, gamma_ppn=gamma_ppn, lambda_mst=lambda_mst,
                                             kappa_ext=kappa_ext, aniso_param=aniso_param)
        return logL
