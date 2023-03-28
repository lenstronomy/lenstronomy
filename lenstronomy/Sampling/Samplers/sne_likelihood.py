from lenstronomy.Sampling.likelihood import LikelihoodModule
from lenstronomy.Cosmo.lens_cosmo import fermat_potential2time_delay


class SNeLikelihood(LikelihoodModule):
    """
    dedicated likelihood class for a multiply imaged lensed SNe model.
    The input extends from the traditional lenstronomy modeling to a SNe light curve model.
    """

    def logL(self, args, verbose=False):
        """

        :param args:
        :param verbose:
        :return:
        """
        kwargs_return = self.param.args2kwargs(args)
        # TODO extract additional SNe parameters
        kwargs_sne = {}

        # compute image magnifications and time delays
        kwargs_ps = kwargs_return['kwargs_ps']
        kwargs_lens = kwargs_return['kwargs_lens']
        x_image_list, y_image_list = self.PointSource.image_position(kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens)
        # the first point-source model is the one we are using as the lensed SNe images
        ra_image, dec_image = x_image_list[0], y_image_list[0]
        image_magnification = self.LensModel.magnification(ra_image, dec_image, kwargs_lens)
        # time delays
        fermat_pot = self.LensModel.fermat_potential(ra_image, dec_image, kwargs_lens)
        ddt = kwargs_return['kwargs_special']['D_dt']
        time_delay = fermat_potential2time_delay(fermat_pot, ddt=ddt, kappa_ext=0)
        # compute flux in different bands at the epochs of the images taken

        # compute image likelihood for all the bands
