from lenstronomy.ImSim.make_image import MakeImage
from astrofunc.util import Util_class


class FluxRatios(object):
    """
    class to compute flux ratio anomalies, inherited from standard MakeImage
    """
    def __init__(self, kwargs_options, kwargs_data, kwargs_psf):
        self.makeImage = MakeImage(kwargs_options, kwargs_data, kwargs_psf=kwargs_psf)
        self.kwargs_data = kwargs_data
        self.kwargs_options = kwargs_options

    def flux_ratios(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, source_size=0.003):

        deltaPix = self.kwargs_data['deltaPix']
        image = self.kwargs_data['image_data']
        numPix = len(image)
        subgrid_res = self.kwargs_options['subgrid_res']

        util_class = Util_class()
        x_grid_sub, y_grid_sub = util_class.make_subgrid(self.kwargs_data['x_coords'], self.kwargs_data['y_coords'],
                                                         subgrid_res)

        model, error_map, cov_param, param = self.makeImage.make_image_ideal(x_grid_sub, y_grid_sub, kwargs_lens,
                                                                        kwargs_source,
                                                                        kwargs_lens_light, kwargs_else, numPix,
                                                                        deltaPix, subgrid_res, inv_bool=True)
        amp_list, _ = self.makeImage.get_image_amplitudes(param, kwargs_else)

        ra_pos, dec_pos, mag = self.makeImage.get_magnification_model(kwargs_lens, kwargs_else)
        mag_finite = self.makeImage.get_magnification_finite(kwargs_lens, kwargs_else, source_sigma=source_size,
                                                             delta_pix=source_size*100, subgrid_res=1000)
        return amp_list, mag, mag_finite
