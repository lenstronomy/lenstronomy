from lenstronomy.Sampling.likelihood import LikelihoodModule
from lenstronomy.Cosmo.lens_cosmo import fermat_potential2time_delay


class SNeLikelihood(LikelihoodModule):
    """
    dedicated likelihood class for a multiply imaged lensed SNe model.
    The input extends from the traditional lenstronomy modeling to a SNe light curve model.
    """

    def __int__(self, lightcurve_function, **kwargs):
        """

        :param lightcurve_function: definition that returns supernova light curve
         (see requirement in sne_amplitude() definition)
        :param kwargs: keyword arguments to initialize LikelihoodModule() class
        :return:
        """
        self._lightcurve_function = lightcurve_function
        super(SNeLikelihood).__init__(self, **kwargs)


    def logL(self, args, verbose=False):
        """

        :param args: parameters being sampled
        :type args: ordered arguments
        :param verbose: if True, prints intermediate results
        :type verbose: boolean
        :return: log likelihood of data given model
        """
        kwargs_return = self.param.args2kwargs(args)
        # extract additional SNe parameters
        kwargs_special = kwargs_return['kwargs_special']
        kwargs_sne = self.param.specialParams.sne_kwargs(kwargs_special)

        # compute image magnifications and time delays
        kwargs_ps = kwargs_return['kwargs_ps']
        kwargs_lens = kwargs_return['kwargs_lens']
        x_image_list, y_image_list = self.PointSource.image_position(kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens)
        # the first point-source model is the one we are using as the lensed SNe images
        ra_image, dec_image = x_image_list[0], y_image_list[0]
        image_magnification = self.LensModel.magnification(ra_image, dec_image, kwargs_lens)
        # time delays
        fermat_pot = self.LensModel.fermat_potential(ra_image, dec_image, kwargs_lens)
        # the user needs to sample or keep fixed 'D_dt', otherwise no time-delay can be computed.
        ddt = kwargs_return['kwargs_special']['D_dt']
        time_delays = fermat_potential2time_delay(fermat_pot, ddt=ddt, kappa_ext=0)
        # compute flux in different bands at the epochs of the images taken

        amp_list = self.sne_amplitudes(image_magnification, time_delays, kwargs_sne)

        # compute image likelihood for all the bands
        # TODO: make a ImageModel class with fixed point sources while letting other parameters being linearly fit for
        return 0

    def sne_amplitudes(self, image_magnification, time_delays, kwargs_sne):
        """
        compute flux in different bands at the epochs of the images taken

        :param image_magnification: macro-lensing magnifications (incl. sign) in the order of the images
        :type image_magnification: numpy array of length of the images
        :param time_delays: predicted time delay of the images relative to a straight line to the source without lensing
        :type time_delays: numpy array of length of the images
        :param kwargs_sne: supernovae parameters for the light curve
        :type kwargs_sne: dictionary
        :return: list of image amplitudes for all the imaging bands
        """
        amp_list = self._lightcurve_function(image_magnification, time_delays, **kwargs_sne)
        return amp_list
