__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.Cosmo.lens_properties import LensProp


class TestParam(object):

    def setup(self):
        pass

    def test_velocity_dispersion_new(self):
        z_lens = 0.5
        z_source = 1.5
        kwargs_data = {}
        kwargs_options = {'lens_model_list': ['NFW_ELLIPSE', 'PJAFFE_ELLIPSE', 'EXTERNAL_SHEAR', 'SIE', 'SIE', 'SIE'],
                         'foreground_shear': False, 'lens_model_internal_bool': [True, True, False, False, False, False],
                          'lens_light_model_internal_bool': [True], 'lens_light_model_list': ['DOUBLE_SERSIC']}
        lensProp = LensProp(z_lens, z_source, kwargs_options, kwargs_data)
        kwargs_lens = [{'theta_Rs': 1.6146238105003521, 'Rs': 5.8215556092277279, 'q': 0.89960421953918235,
                        'center_x': -0.065699948429581406, 'center_y': 0.024186165973141851, 'phi_G': -0.293898099102263},
                       {'Rs': 3.3343851394796515, 'q': 0.71808978862755635, 'center_x': 0, 'center_y': 0, 'Ra': 1.3113719625383848,
                        'phi_G': 1.2119791682610044, 'sigma0': 0.3405563712096914}, {'e1': -0.050871696555354479,
                        'e2': -0.0061601733920590464}, {'q': 0.28297894544102642, 'center_y': 2.79985456, 'center_x': -2.32019894,
                        'theta_E': 0.28165274714097904, 'phi_G': 0.483913071449317}, {'q': 0.18042017015594097, 'center_y': 3.83985426,
                        'center_x': -2.32019933, 'theta_E': 0.0038110812674654873, 'phi_G': 1.4035952823380935},
                       {'q': 0.50081325034773538, 'center_y': 4.31985428, 'center_x': -1.68019931, 'theta_E': 0.45552039839735037,
                        'phi_G': -1.1893496113990851}]
        kwargs_lens_light = [{'n_sersic': 1.1212528655709217, 'R_2': 0.03025682660635394, 'I0_2': 139.96763298885992,
                              'center_x': -0.019674496231393473, 'n_2': 0.20000008624093865, 'smoothing': 0.01,
                              'q': 0.79703498156919605, 'I0_sersic': 1.1091367792010356, 'center_y': 0.076914975081560991,
                              'phi_G': -0.52624727893702705, 'R_sersic': 0.42691611878867058}]
        kwargs_anisotropy = {'r_ani': 0.62737495790281372}
        kwargs_aperture = {'center_ra': 0, 'width': 1.0, 'length': 3.8, 'angle': 0, 'center_dec': 0}
        aperture_type = 'slit'
        psf_fwhm = 0.7
        anisotropy_model = 'OsipkovMerritt'
        r_eff = 0.211919902322
        v_sigma = lensProp.velocity_disperson_new(kwargs_lens, kwargs_lens_light, kwargs_anisotropy, kwargs_aperture, psf_fwhm, aperture_type, anisotropy_model, MGE_light=True, r_eff=r_eff)
        npt.assert_almost_equal(v_sigma, 171.65866292, decimal=1)


if __name__ == '__main__':
    pytest.main()