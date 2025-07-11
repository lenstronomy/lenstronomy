import pytest
import numpy as np

from lenstronomy.SourceReconstruction.pixelated_source_reconstruction import (
    PixelatedSourceReconstruction,
)
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Data.pixel_grid import PixelGrid


class TestPixelatedSourceReconstruction(object):
    def setup_method(self):
        # Test with a 10x10 random data image and a 19x19 pixelized gaussian PSF
        self.image_data = np.array(
            [
                [
                    0.37998374,
                    0.45868146,
                    0.94466124,
                    0.49985,
                    0.61152472,
                    0.90291398,
                    0.93014262,
                    0.16425493,
                    0.46615874,
                    0.37820467,
                ],
                [
                    0.20168678,
                    0.62396235,
                    0.2321545,
                    0.09974247,
                    0.244949,
                    0.18138219,
                    0.80001545,
                    0.80389765,
                    0.98861931,
                    0.14030475,
                ],
                [
                    0.27689752,
                    0.75104745,
                    0.65846718,
                    0.07815372,
                    0.09323168,
                    0.86884209,
                    0.51337759,
                    0.05514285,
                    0.21899282,
                    0.89654578,
                ],
                [
                    0.43021229,
                    0.53728807,
                    0.62681958,
                    0.62799635,
                    0.5783872,
                    0.20735348,
                    0.52720995,
                    0.71356574,
                    0.62004311,
                    0.52865161,
                ],
                [
                    0.75981885,
                    0.25822367,
                    0.71744285,
                    0.97587512,
                    0.20702018,
                    0.91744571,
                    0.42387492,
                    0.36172342,
                    0.07509998,
                    0.14109497,
                ],
                [
                    0.71996296,
                    0.43829422,
                    0.4490221,
                    0.4027239,
                    0.81514223,
                    0.53845533,
                    0.49095991,
                    0.14225506,
                    0.37623786,
                    0.90357779,
                ],
                [
                    0.24835311,
                    0.9566734,
                    0.26708036,
                    0.42091185,
                    0.08642987,
                    0.92327661,
                    0.92215173,
                    0.79882988,
                    0.64493301,
                    0.0992427,
                ],
                [
                    0.51896031,
                    0.57795715,
                    0.64121395,
                    0.84125236,
                    0.94663785,
                    0.08461582,
                    0.94735046,
                    0.28469182,
                    0.06974356,
                    0.35807553,
                ],
                [
                    0.14619233,
                    0.0142298,
                    0.81142352,
                    0.36906236,
                    0.14258796,
                    0.91290133,
                    0.04122539,
                    0.50850401,
                    0.88442304,
                    0.15904894,
                ],
                [
                    0.51761851,
                    0.39558722,
                    0.40555487,
                    0.51119774,
                    0.70308528,
                    0.70418488,
                    0.23706343,
                    0.94956522,
                    0.64892141,
                    0.99079498,
                ],
            ]
        )
        self.numPix = self.image_data.shape[0]
        self.deltaPix = 0.05
        self.background_rms = 1
<<<<<<< HEAD
        self.kwargs_data = sim_util.data_configure_simple(self.numPix, self.deltaPix, np.inf, self.background_rms)
        self.kwargs_data['image_data'] = self.image_data
        self.kwargs_data['ra_at_xy_0'] = -self.numPix * self.deltaPix / 2
        self.kwargs_data['dec_at_xy_0'] = -self.numPix * self.deltaPix / 2
        self.data_class = ImageData(**self.kwargs_data)
        
        self.lens_model_list = ['SIE']
        self.lens_model_class = LensModel(lens_model_list=self.lens_model_list)
        self.kwargs_lens = [{'theta_E': 0.2, 'e1': 0.0, 'e2': 0.0, 'center_x': 0.0, 'center_y': 0.0}]
        
        gaussian_kernel = np.zeros((19,19))
=======
        self.kwargs_data = sim_util.data_configure_simple(
            self.numPix, self.deltaPix, np.inf, self.background_rms
        )
        self.kwargs_data["image_data"] = self.image_data
        self.kwargs_data["ra_at_xy_0"] = -self.numPix * self.deltaPix / 2
        self.kwargs_data["dec_at_xy_0"] = -self.numPix * self.deltaPix / 2
        self.lens_model_list = ["SIE"]
        self.kwargs_lens = [
            {"theta_E": 0.2, "e1": 0.0, "e2": 0.0, "center_x": 0.0, "center_y": 0.0}
        ]

        # PSF kernel must be odd and >= 2 * numPix - 1
        gaussian_kernel = np.zeros((19, 19))
>>>>>>> 8389a9906b9db3877f2bc0296b72e90037f4489e
        sigma = 3
        for i in range(19):
            for j in range(19):
                gaussian_kernel[i, j] = np.exp(
                    -0.5 * ((9 - i) ** 2 + (9 - j) ** 2) / sigma**2
                )
        gaussian_kernel /= gaussian_kernel.sum()
        self.kernel = gaussian_kernel
        self.kwargs_psf = {'psf_type': 'PIXEL', 'pixel_size': self.deltaPix,
                           'kernel_point_source': self.kernel, 'kernel_point_source_normalisation': False}
        self.psf_class = PSF(**self.kwargs_psf)
        
        self.source_pixel_width = 0.03
        transform_pix2angle = self.source_pixel_width * np.identity(2)
        self.kwargs_source_grid = {'nx': 2, 'ny': 3, 'transform_pix2angle': transform_pix2angle,
                                   'ra_at_xy_0': -2.0 * 0.03, 'dec_at_xy_0': -2.0 * 0.03}
        self.source_pixel_grid_class = PixelGrid(**self.kwargs_source_grid)

    def test_init(self):
<<<<<<< HEAD
        psr = PixelatedSourceReconstruction(self.data_class, self.psf_class, self.lens_model_class, 
                                            self.source_pixel_grid_class, kwargs_lens=self.kwargs_lens)
=======
        psr = PixelatedSourceReconstruction(
            self.kwargs_data,
            self.lens_model_list,
            kwargs_lens=self.kwargs_lens,
            kernel=self.kernel,
        )
>>>>>>> 8389a9906b9db3877f2bc0296b72e90037f4489e
        assert psr._numPix == len(self.image_data)
        assert np.allclose(psr._image_data, self.image_data)
        assert np.allclose(psr._noise_rms, self.background_rms)
        assert psr._primary_beam is None
        assert psr._logL_method == "diagonal"
        assert psr._verbose is False
<<<<<<< HEAD
        assert np.allclose(psr._nx_source, self.kwargs_source_grid['nx'])
        assert np.allclose(psr._ny_source, self.kwargs_source_grid['ny'])
        assert np.allclose(psr._pixel_width_source, self.kwargs_source_grid['transform_pix2angle'][0,0])
        assert np.allclose(psr._kernel, self.kernel)
        
        # Test verbose = True
        psr = PixelatedSourceReconstruction(self.data_class, self.psf_class, self.lens_model_class, 
                                            self.source_pixel_grid_class, kwargs_lens=self.kwargs_lens, verbose=True)
        assert psr._verbose is True
        
        # Test invalid source grid input
        source_grid_invalid1 = PixelGrid(nx = 3, ny = 3, transform_pix2angle = np.array(((1., 0.1), (0., 1.))),
                                ra_at_xy_0 = 0, dec_at_xy_0 = 0)
        source_grid_invalid2 = PixelGrid(nx = 3, ny = 3, transform_pix2angle = np.array(((1., 0.), (0.1, 1.))),
                                ra_at_xy_0 = 0, dec_at_xy_0 = 0)
        source_grid_invalid3 = PixelGrid(nx = 3, ny = 3, transform_pix2angle = np.array(((1., 0.), (0., 1.1))),
                                ra_at_xy_0 = 0, dec_at_xy_0 = 0)
        with pytest.raises(ValueError):
            PixelatedSourceReconstruction(self.data_class, self.psf_class, self.lens_model_class,
                                          source_grid_invalid1, kwargs_lens=self.kwargs_lens)
        with pytest.raises(ValueError):
            PixelatedSourceReconstruction(self.data_class, self.psf_class, self.lens_model_class,
                                          source_grid_invalid2, kwargs_lens=self.kwargs_lens)
        with pytest.raises(ValueError):
            PixelatedSourceReconstruction(self.data_class, self.psf_class, self.lens_model_class,
                                          source_grid_invalid3, kwargs_lens=self.kwargs_lens)
        
        # Test invalid kernel size for likelihood_method = 'interferometry_natwt'
        kwargs_data_natwt_kernel_test = self.kwargs_data.copy()
        kwargs_data_natwt_kernel_test['likelihood_method'] = 'interferometry_natwt'
        data_class_natwt_kernel_test = ImageData(**kwargs_data_natwt_kernel_test)
        kwargs_psf_natwt_kernel_test1 = self.kwargs_psf.copy()
        kwargs_psf_natwt_kernel_test1['kernel_point_source'] = np.ones((17, 17))
        psf_class_natwt_kernel_test1 = PSF(**kwargs_psf_natwt_kernel_test1)
=======

        # Test verbose = True
        psr = PixelatedSourceReconstruction(
            self.kwargs_data,
            self.lens_model_list,
            kwargs_lens=self.kwargs_lens,
            kernel=self.kernel,
            verbose=True,
        )
        assert psr._verbose is True

        # Test primary_beam input
        kwargs_data_pb = self.kwargs_data.copy()
        kwargs_data_pb["antenna_primary_beam"] = np.random.rand(
            self.numPix, self.numPix
        )
        psr_pb = PixelatedSourceReconstruction(
            kwargs_data_pb, self.lens_model_list, self.kwargs_lens, self.kernel
        )
        assert np.allclose(psr_pb._primary_beam, kwargs_data_pb["antenna_primary_beam"])

        # Test invalid kernel size
        with pytest.raises(ValueError):
            PixelatedSourceReconstruction(
                self.kwargs_data,
                self.lens_model_list,
                self.kwargs_lens,
                np.ones((4, 4)),
            )  # Even dimension
        with pytest.raises(ValueError):
            PixelatedSourceReconstruction(
                self.kwargs_data,
                self.lens_model_list,
                self.kwargs_lens,
                np.ones((17, 17)),
            )  # Too small

        # Test unsupported likelihood method
        kwargs_data_invalid_method = self.kwargs_data.copy()
        kwargs_data_invalid_method["likelihood_method"] = "WRONG"
        with pytest.raises(ValueError):
            PixelatedSourceReconstruction(
                kwargs_data_invalid_method,
                self.lens_model_list,
                self.kwargs_lens,
                self.kernel,
            )

    def test_generate_M_b_input_validation(self):
        psr = PixelatedSourceReconstruction(
            self.kwargs_data, self.lens_model_list, self.kwargs_lens, self.kernel
        )

        # Test non-integer coordinates
        with pytest.raises(TypeError):
            psr.generate_M_b(0.5, 2, 0, 2)
        with pytest.raises(TypeError):
            psr.generate_M_b(0, 2.0, 0, 2)
        with pytest.raises(TypeError):
            psr.generate_M_b(0, 2, 0.5, 2)
        with pytest.raises(TypeError):
            psr.generate_M_b(0, 2, 0, 2.0)

        # Test out of bounds coordinates
        with pytest.raises(ValueError):
            psr.generate_M_b(-1, 2, 0, 2)
        with pytest.raises(ValueError):
            psr.generate_M_b(0, self.numPix + 1, 0, 2)
        with pytest.raises(ValueError):
            psr.generate_M_b(2, 0, 0, 2)
        with pytest.raises(ValueError):
            psr.generate_M_b(2, 2, 0, 2)
        with pytest.raises(ValueError):
            psr.generate_M_b(0, 2, -1, 2)
        with pytest.raises(ValueError):
            psr.generate_M_b(0, 2, 0, self.numPix + 1)
        with pytest.raises(ValueError):
            psr.generate_M_b(0, 2, 2, 0)
>>>>>>> 8389a9906b9db3877f2bc0296b72e90037f4489e
        with pytest.raises(ValueError):
            PixelatedSourceReconstruction(data_class_natwt_kernel_test, psf_class_natwt_kernel_test1, 
                                          self.lens_model_class, self.source_pixel_grid_class, 
                                          kwargs_lens=self.kwargs_lens)
    
    def test_generate_M_b(self):
        psr = PixelatedSourceReconstruction(self.data_class, self.psf_class, self.lens_model_class, 
                                            self.source_pixel_grid_class, kwargs_lens=self.kwargs_lens)
        M_default, b_default = psr.generate_M_b()
        M_diagonal, b_diagonal = psr.generate_M_b_diagonal_likelihood()
        assert np.allclose(M_default, M_diagonal)
        assert np.allclose(b_default, b_diagonal)
        
        kwargs_data_interferometry_likelihood_test = self.kwargs_data.copy()
        kwargs_data_interferometry_likelihood_test['likelihood_method'] = 'interferometry_natwt'
        data_class_interferometry_likelihood_test = ImageData(**kwargs_data_interferometry_likelihood_test)
        psr_natwt = PixelatedSourceReconstruction(data_class_interferometry_likelihood_test, self.psf_class, 
                                                  self.lens_model_class, self.source_pixel_grid_class, 
                                                  kwargs_lens=self.kwargs_lens)
        M_natwt0, b_natwt0 = psr_natwt.generate_M_b()
        M_natwt1, b_natwt1 = psr_natwt.generate_M_b_interferometry_natwt_likelihood()
        assert np.allclose(M_natwt0, M_natwt1)
        assert np.allclose(b_natwt0, b_natwt1)
        
    def test_generate_M_b_diagonal_likelihood(self):
<<<<<<< HEAD
        psr = PixelatedSourceReconstruction(self.data_class, self.psf_class, self.lens_model_class, 
                                            self.source_pixel_grid_class, kwargs_lens=self.kwargs_lens)
        M_result, b_result = psr.generate_M_b_diagonal_likelihood()
        M_expected = np.array([[0.01129016, 0.01358918, 0.01358918, 0.02152248, 0.01484909, 0.02400253],
                               [0.01358918, 0.01815592, 0.01604542, 0.02759236, 0.01819409, 0.03177232],
                               [0.01358918, 0.01604542, 0.01815592, 0.02759236, 0.02200811, 0.03442514],
                               [0.02152248, 0.02759236, 0.02759236, 0.04546299, 0.03301198, 0.0554495 ],
                               [0.01484909, 0.01819409, 0.02200811, 0.03301198, 0.03050647, 0.04755653],
                               [0.02400253, 0.03177232, 0.03442514, 0.0554495 , 0.04755653, 0.07859336]])
        b_expected = np.array([0.44319784, 0.60691129, 0.60445714, 1.00888506, 0.80968046, 1.37626347])
        assert np.allclose(M_expected, M_result, atol = 1e-5)
        assert np.allclose(b_expected, b_result, atol = 1e-5)
        
    def test_generate_M_b_interferometry_natwt_likelihood(self):
        kwargs_data_interferometry = self.kwargs_data.copy()
        kwargs_data_interferometry['likelihood_method'] = 'interferometry_natwt'
=======
        psr = PixelatedSourceReconstruction(
            self.kwargs_data, self.lens_model_list, self.kwargs_lens, self.kernel
        )
        M_result, b_result = psr.generate_M_b_diagonal_likelihood(3, 6, 3, 6)
        M_expected = np.array(
            [
                [
                    0.01589136,
                    0.01418308,
                    0.01166159,
                    0.01418308,
                    0.0435946,
                    0.05707797,
                    0.01166159,
                    0.05707798,
                    0.15287013,
                ],
                [
                    0.01418308,
                    0.01296342,
                    0.01091549,
                    0.0125361,
                    0.03944186,
                    0.05427183,
                    0.01067403,
                    0.05110745,
                    0.14155157,
                ],
                [
                    0.01166159,
                    0.01091549,
                    0.01486035,
                    0.01067403,
                    0.03498845,
                    0.05898625,
                    0.01383971,
                    0.05793957,
                    0.16813556,
                ],
                [
                    0.01418308,
                    0.0125361,
                    0.01067403,
                    0.01296342,
                    0.03944186,
                    0.05110745,
                    0.01091549,
                    0.05427183,
                    0.14155157,
                ],
                [
                    0.0435946,
                    0.03944186,
                    0.03498845,
                    0.03944186,
                    0.12289403,
                    0.16808722,
                    0.03498845,
                    0.16808722,
                    0.45495481,
                ],
                [
                    0.05707797,
                    0.05427183,
                    0.05898625,
                    0.05110745,
                    0.16808722,
                    0.28053797,
                    0.05793957,
                    0.25100395,
                    0.73609751,
                ],
                [
                    0.01166159,
                    0.01067403,
                    0.01383971,
                    0.01091549,
                    0.03498845,
                    0.05793957,
                    0.01486035,
                    0.05898625,
                    0.16813556,
                ],
                [
                    0.05707798,
                    0.05110745,
                    0.05793957,
                    0.05427183,
                    0.16808722,
                    0.25100395,
                    0.05898625,
                    0.28053798,
                    0.73609751,
                ],
                [
                    0.15287013,
                    0.14155157,
                    0.16813556,
                    0.14155157,
                    0.45495481,
                    0.73609751,
                    0.16813556,
                    0.73609751,
                    2.06919029,
                ],
            ]
        )
        b_expected = np.array(
            [
                0.55025103,
                0.50942389,
                0.57601736,
                0.50640756,
                1.62382074,
                2.60211843,
                0.56026027,
                2.56493507,
                7.17248711,
            ]
        )
        assert np.allclose(M_expected, M_result, atol=1e-5)
        assert np.allclose(b_expected, b_result, atol=1e-5)

    def test_generate_M_b_interferometry_natwt_likelihood(self):
        kwargs_data_interferometry = self.kwargs_data.copy()
        kwargs_data_interferometry["likelihood_method"] = "interferometry_natwt"
>>>>>>> 8389a9906b9db3877f2bc0296b72e90037f4489e

        # Test with a gaussian primary beam
        gaussian_PB = np.zeros((10, 10))
        sigma = 5
        for i in range(10):
            for j in range(10):
                gaussian_PB[i, j] = np.exp(
                    -0.5 * ((3 - i) ** 2 + (3 - j) ** 2) / sigma**2
                )
        gaussian_PB /= gaussian_PB.max()
        primary_beam = gaussian_PB
<<<<<<< HEAD
        kwargs_data_interferometry['antenna_primary_beam'] = primary_beam
        
        data_class_interferometry = ImageData(**kwargs_data_interferometry)
        
        # Note that the PSF here used is not the real dirty beam of the interferometric images
        # Here we just use a gaussian PSF to test the output of the function
        psr = PixelatedSourceReconstruction(data_class_interferometry, self.psf_class, self.lens_model_class, 
                                            self.source_pixel_grid_class, kwargs_lens=self.kwargs_lens)
        M_result, b_result = psr.generate_M_b_interferometry_natwt_likelihood()
        M_expected = np.array([[0.02229289, 0.0221506 , 0.0221506 , 0.03531318, 0.01467593, 0.02430386],
                               [0.0221506 , 0.03126001, 0.01790875, 0.03867838, 0.01132409, 0.02526159],
                               [0.0221506 , 0.01790875, 0.03126001, 0.03867838, 0.02907649, 0.03816851],
                               [0.03531318, 0.03867838, 0.03867838, 0.06318939, 0.03045054, 0.05084672],
                               [0.01467593, 0.01132409, 0.02907649, 0.03045054, 0.040716  , 0.04901641],
                               [0.02430386, 0.02526159, 0.03816851, 0.05084672, 0.04901641, 0.06995898]])
        b_expected = np.array([0.84629238, 0.8555345 , 0.93490381, 1.09044392, 1.06591831, 1.55618881])
        assert np.allclose(M_expected, M_result, atol = 1e-5)
        assert np.allclose(b_expected, b_result, atol = 1e-5)

    def test_lens_pixel_source_of_a_rectangular_region(self):
        psr = PixelatedSourceReconstruction(self.data_class, self.psf_class, self.lens_model_class, 
                                            self.source_pixel_grid_class, kwargs_lens=self.kwargs_lens)
        lensed_pixels = psr.lens_pixel_source_of_a_rectangular_region()
        assert len(lensed_pixels) == 6
        assert type(lensed_pixels[3][0][0]) == int
        assert type(lensed_pixels[3][0][1]) == int
        assert np.allclose(lensed_pixels[3][0][2], 0.07323579271979347, atol = 1e-5)
        
        # test when there is no lens and the source grid is right on the image grid
        kwargs_source_grid_1 = {'nx': 3, 'ny': 3, 'transform_pix2angle': 0.05 * np.identity(2),
                                   'ra_at_xy_0': -2.0 * 0.05, 'dec_at_xy_0': -2.0 * 0.05}
        source_pixel_grid_class_1 = PixelGrid(**kwargs_source_grid_1)
        psr_no_lens = PixelatedSourceReconstruction(self.data_class, self.psf_class, self.lens_model_class, source_pixel_grid_class_1,
                            kwargs_lens = [{'theta_E': 0.0, 'e1': 0.0, 'e2': 0.0, 'center_x': 0.0, 'center_y': 0.0}])
        lensed_pixels_no_lens = psr_no_lens.lens_pixel_source_of_a_rectangular_region()
=======
        kwargs_data_interferometry["antenna_primary_beam"] = primary_beam

        # Note that the PSF here used is not the real dirty beam of the interferometric images
        # Here we just use a gaussian PSF to test the output of the function

        psr = PixelatedSourceReconstruction(
            kwargs_data_interferometry,
            self.lens_model_list,
            self.kwargs_lens,
            self.kernel,
        )
        M_result, b_result = psr.generate_M_b_diagonal_likelihood(3, 6, 3, 6)
        M_expected = np.array(
            [
                [
                    0.00844248,
                    0.00781839,
                    0.00575649,
                    0.00781839,
                    0.02456662,
                    0.02928061,
                    0.00575649,
                    0.02928061,
                    0.07914203,
                ],
                [
                    0.00781839,
                    0.00743985,
                    0.00534869,
                    0.00715349,
                    0.02305478,
                    0.02859512,
                    0.00523623,
                    0.0266021,
                    0.07443979,
                ],
                [
                    0.00575649,
                    0.00534869,
                    0.00679656,
                    0.00523623,
                    0.01693452,
                    0.02649154,
                    0.00632314,
                    0.02610573,
                    0.07504186,
                ],
                [
                    0.00781839,
                    0.00715349,
                    0.00523623,
                    0.00743985,
                    0.02305478,
                    0.0266021,
                    0.00534869,
                    0.02859512,
                    0.07443979,
                ],
                [
                    0.02456662,
                    0.02305478,
                    0.01693452,
                    0.02305478,
                    0.073205,
                    0.08879091,
                    0.01693452,
                    0.08879091,
                    0.23986924,
                ],
                [
                    0.02928061,
                    0.02859512,
                    0.02649154,
                    0.0266021,
                    0.08879091,
                    0.13382554,
                    0.02610573,
                    0.11703491,
                    0.34634772,
                ],
                [
                    0.00575649,
                    0.00523623,
                    0.00632314,
                    0.00534869,
                    0.01693452,
                    0.02610573,
                    0.00679656,
                    0.02649154,
                    0.07504186,
                ],
                [
                    0.02928061,
                    0.0266021,
                    0.02610573,
                    0.02859512,
                    0.08879091,
                    0.11703491,
                    0.02649154,
                    0.13382555,
                    0.34634773,
                ],
                [
                    0.07914203,
                    0.07443979,
                    0.07504186,
                    0.07443979,
                    0.23986924,
                    0.34634772,
                    0.07504186,
                    0.34634773,
                    0.96454003,
                ],
            ]
        )
        b_expected = np.array(
            [
                0.39742724,
                0.37285967,
                0.39162244,
                0.37110292,
                1.19387532,
                1.76438593,
                0.38110624,
                1.74332944,
                4.8621466,
            ]
        )
        assert np.allclose(M_expected, M_result, atol=1e-5)
        assert np.allclose(b_expected, b_result, atol=1e-5)

    def test_lens_pixel_source_of_a_rectangular_region(self):
        psr = PixelatedSourceReconstruction(
            self.kwargs_data, self.lens_model_list, self.kwargs_lens, self.kernel
        )
        lensed_pixels = psr.lens_pixel_source_of_a_rectangular_region(3, 6, 3, 6)
        assert len(lensed_pixels) == 9
        assert type(lensed_pixels[3][0][0]) == int
        assert type(lensed_pixels[3][0][1]) == int
        assert np.allclose(lensed_pixels[3][0][2], 0.061888246868462754, atol=1e-5)

        psr_no_lens = PixelatedSourceReconstruction(
            self.kwargs_data,
            self.lens_model_list,
            [{"theta_E": 0.0, "e1": 0.0, "e2": 0.0, "center_x": 0.0, "center_y": 0.0}],
            self.kernel,
        )
        lensed_pixels_no_lens = psr_no_lens.lens_pixel_source_of_a_rectangular_region(
            3, 6, 3, 6
        )
>>>>>>> 8389a9906b9db3877f2bc0296b72e90037f4489e
        assert len(lensed_pixels_no_lens) == 9
        assert np.allclose(lensed_pixels_no_lens[3][0][2], 0.0, atol=1e-5)
        assert np.allclose(lensed_pixels_no_lens[3][1][2], 0.0, atol=1e-5)
        assert np.allclose(lensed_pixels_no_lens[3][2][2], 0.0, atol=1e-5)
        assert np.allclose(lensed_pixels_no_lens[3][3][2], 1.0, atol=1e-5)

    def test_lens_an_image_by_rayshooting(self):
<<<<<<< HEAD
        psr = PixelatedSourceReconstruction(self.data_class, self.psf_class, self.lens_model_class, 
                                            self.source_pixel_grid_class, kwargs_lens=self.kwargs_lens)
        image_lensed = psr.lens_an_image_by_rayshooting(self.image_data[1:4,1:3])
        image_lensed_expected = np.array([[0.        , 0.        , 0.        , 0.        , 0.11461723,
                                              0.        , 0.        , 0.        , 0.        , 0.        ],
                                             [0.        , 0.61209186, 0.51636297, 0.22842217, 0.03150559,
                                              0.        , 0.        , 0.        , 0.        , 0.        ],
                                             [0.        , 0.68932727, 0.18182989, 0.        , 0.        ,
                                              0.        , 0.        , 0.10354675, 0.        , 0.        ],
                                             [0.        , 0.44900584, 0.        , 0.        , 0.        ,
                                              0.        , 0.        , 0.        , 0.21783726, 0.        ],
                                             [0.61899174, 0.1250819 , 0.        , 0.        , 0.        ,
                                              0.        , 0.        , 0.        , 0.33383885, 0.        ],
                                             [0.56713191, 0.        , 0.        , 0.        , 0.        ,
                                              0.        , 0.        , 0.        , 0.5671319 , 1.18399251e-08],
                                             [0.35597634, 0.11855964, 0.        , 0.        , 0.        ,
                                              0.        , 0.        , 0.        , 0.63783075, 0.        ],
                                             [0.        , 0.28591644, 0.        , 0.        , 0.        ,
                                              0.        , 0.        , 0.57478178, 0.34918442, 0.        ],
                                             [0.        , 2.00583843e-08 , 0.12798694, 0.        , 0.        ,
                                              0.        , 0.229636  , 0.2348939 , 0.        , 0.        ],
                                             [0.        , 0.        , 0.        , 0.06533276, 0.02498332,
                                              0.        , 0.        , 0.        , 0.        , 0.        ]])
        assert np.allclose(image_lensed, image_lensed_expected, atol = 1e-5)
        
        # Test lensing an image when there is no lens and the source grid is right on the image grid
        # The result should be the original image
        kwargs_source_grid_1 = {'nx': 3, 'ny': 3, 'transform_pix2angle': 0.05 * np.identity(2),
                                   'ra_at_xy_0': -2.0 * 0.05, 'dec_at_xy_0': -2.0 * 0.05}
        source_pixel_grid_class_1 = PixelGrid(**kwargs_source_grid_1)
        psr_no_lens = PixelatedSourceReconstruction(self.data_class, self.psf_class, self.lens_model_class, source_pixel_grid_class_1,
                            kwargs_lens = [{'theta_E': 0.0, 'e1': 0.0, 'e2': 0.0, 'center_x': 0.0, 'center_y': 0.0}])
        image_lensed_no_lense = psr_no_lens.lens_an_image_by_rayshooting(self.image_data[3:6, 3:6])
        assert np.allclose(self.image_data[3:6,3:6], image_lensed_no_lense[3:6,3:6], atol = 1e-5)
        
        with pytest.raises(ValueError):
            psr.lens_an_image_by_rayshooting(np.random.rand(9, 9))
            
    def test_sparse_matrix_manipulation_functions(self):
        
        # Define a source reconstrution with a smaller data image size
        kwargs_data_small = sim_util.data_configure_simple(5, 0.05, np.inf, 1)
        data_class_small = ImageData(**kwargs_data_small)
        psr_small = PixelatedSourceReconstruction(data_class_small, self.psf_class, self.lens_model_class, 
                                                  self.source_pixel_grid_class, kwargs_lens=self.kwargs_lens)
        
        # Define a source reconstrution with a smaller data image size 
        # AND a small kernel (kernel size is smaller than the image)
        kernel_small = self.kernel[8:11,8:11]
        kwargs_psf_small_kernel = self.kwargs_psf.copy()
        kwargs_psf_small_kernel['kernel_point_source'] = kernel_small
        psf_class_small_kernel = PSF(**kwargs_psf_small_kernel)
        psr_small_kernel = PixelatedSourceReconstruction(data_class_small, psf_class_small_kernel, self.lens_model_class, 
                                                  self.source_pixel_grid_class, kwargs_lens=self.kwargs_lens)
        
        sp1 = [[0, 0, 0.6862915211329877],
               [0, 1, 0.333126273654468],
               [1, 0, 0.3331262933721124],
               [1, 1, 0.029437248042628716],
               [2, 2, 0.0]]
        sp2 = [[1, 0, 0.12198066661652618],
               [2, 0, 0.9999999906666668],
               [3, 0, 0.12198066661652593]]
        
        
=======
        psr = PixelatedSourceReconstruction(
            self.kwargs_data, self.lens_model_list, self.kwargs_lens, self.kernel
        )
        image_lensed = psr.lens_an_image_by_rayshooting(self.image_data)
        image_lensed_expected = np.array(
            [
                [
                    0.55057385,
                    0.60191051,
                    0.42415815,
                    0.5079897,
                    0.7272718,
                    0.9174457,
                    0.76952424,
                    0.58009972,
                    0.48451217,
                    0.59247616,
                ],
                [
                    0.7510897,
                    0.38147871,
                    0.53384581,
                    0.66890504,
                    0.5884594,
                    0.53845534,
                    0.5807094,
                    0.6486969,
                    0.5990342,
                    0.43825627,
                ],
                [
                    0.6415066,
                    0.51338512,
                    0.62189388,
                    0.657692,
                    0.84146608,
                    0.92327659,
                    0.68314673,
                    0.64532984,
                    0.58219963,
                    0.51373705,
                ],
                [
                    0.55816777,
                    0.64730412,
                    0.61039638,
                    0.84972894,
                    0.8315726,
                    0.08461584,
                    0.55284693,
                    0.32225796,
                    0.63349989,
                    0.5586463,
                ],
                [
                    0.67181196,
                    0.57929582,
                    0.61241512,
                    0.73237099,
                    0.47072161,
                    0.91290133,
                    0.77426614,
                    0.34242006,
                    0.62585474,
                    0.542508,
                ],
                [
                    0.81514223,
                    0.53845533,
                    0.49095991,
                    0.14225506,
                    0.37623786,
                    0.53845533,
                    0.4490221,
                    0.4027239,
                    0.81514223,
                    0.53845533,
                ],
                [
                    0.63854811,
                    0.57901911,
                    0.50719403,
                    0.36718019,
                    0.62854092,
                    0.86884208,
                    0.6579987,
                    0.63550422,
                    0.65092598,
                    0.5444381,
                ],
                [
                    0.4322019,
                    0.63726295,
                    0.5737577,
                    0.50693701,
                    0.48856537,
                    0.20735349,
                    0.43964827,
                    0.40418929,
                    0.64130248,
                    0.60378638,
                ],
                [
                    0.29466049,
                    0.45620148,
                    0.61917282,
                    0.62041542,
                    0.73315565,
                    0.91744569,
                    0.70512864,
                    0.65249452,
                    0.59769631,
                    0.75360961,
                ],
                [
                    0.61751721,
                    0.27845775,
                    0.47782835,
                    0.66009504,
                    0.58870498,
                    0.53845534,
                    0.58316178,
                    0.69506815,
                    0.84007295,
                    0.88943987,
                ],
            ]
        )
        assert np.allclose(image_lensed, image_lensed_expected, atol=1e-5)

        # Test lensing an image when there is no lens
        # The result should be the original image
        psr_no_lens = PixelatedSourceReconstruction(
            self.kwargs_data,
            self.lens_model_list,
            [{"theta_E": 0.0, "e1": 0.0, "e2": 0.0, "center_x": 0.0, "center_y": 0.0}],
            self.kernel,
        )
        image_lensed_no_lense = psr_no_lens.lens_an_image_by_rayshooting(
            self.image_data
        )
        assert np.allclose(self.image_data, image_lensed_no_lense, atol=1e-5)

        with pytest.raises(ValueError):
            psr.lens_an_image_by_rayshooting(np.random.rand(9, 9))

    def test_sparse_matrix_manipulation_functions(self):

        kwargs_data_small = sim_util.data_configure_simple(5, 0.05, np.inf, 1)
        psr_small = PixelatedSourceReconstruction(
            kwargs_data_small, self.lens_model_list, self.kwargs_lens, self.kernel
        )

        sp1 = [
            [0, 0, 0.6862915211329877],
            [0, 1, 0.333126273654468],
            [1, 0, 0.3331262933721124],
            [1, 1, 0.029437248042628716],
            [2, 2, 0.0],
        ]
        sp2 = [
            [1, 0, 0.12198066661652618],
            [2, 0, 0.9999999906666668],
            [3, 0, 0.12198066661652593],
        ]

>>>>>>> 8389a9906b9db3877f2bc0296b72e90037f4489e
        # Test sparse_to_array
        array1 = psr_small.sparse_to_array(sp1)
        array1_expected = np.array(
            [
                [0.68629152, 0.33312627, 0.0, 0.0, 0.0],
                [0.33312629, 0.02943725, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        assert np.allclose(array1, array1_expected, atol=1e-5)

        # Test sum_sparse_elementwise_product
        array2 = psr_small.sparse_to_array(sp2)
        product = psr_small.sum_sparse_elementwise_product(sp1, array2)
        assert np.allclose(product, 0.04063496846014683, atol=1e-5)

        # Test sparse_convolve_and_dot_product
<<<<<<< HEAD
        convolution_product_default = psr_small.sparse_convolve_and_dot_product(sp1, sp2)
        convolution_product = psr_small.sparse_convolve_and_dot_product(sp1, sp2, self.kernel)
        assert np.allclose(convolution_product, 0.025047763176837278, atol = 1e-5)
        assert np.allclose(convolution_product_default, 0.025047763176837278, atol = 1e-5)
        
        # Test sparse_convolve_and_dot_product (when the kernel size is small)
        convolution_product1 = psr_small_kernel.sparse_convolve_and_dot_product([[1,1,1.]],[[3,3,1.5]])
        convolution_product2 = psr_small_kernel.sparse_convolve_and_dot_product([[1,1,1.]],[[2,2,1.5]])
        assert np.allclose(convolution_product1, 0, atol = 1e-5)
        assert np.allclose(convolution_product2, 0.02380609836219314, atol = 1e-5)
        
=======
        convolution_product = psr_small.sparse_convolve_and_dot_product(
            sp1, sp2, self.kernel
        )
        assert np.allclose(convolution_product, 0.025047763176837278, atol=1e-5)

>>>>>>> 8389a9906b9db3877f2bc0296b72e90037f4489e
        # Test sparse_convolution
        sp1_convolved_default = psr_small.sparse_convolution(sp1)
        sp1_convolved = psr_small.sparse_convolution(sp1, self.kernel)
<<<<<<< HEAD
        sp1_convolved_expected = np.array([[0.02381713, 0.0232033 , 0.02027804, 0.01589901, 0.01118485],
                                           [0.0232033 , 0.02259202, 0.01973165, 0.01546063, 0.01086915],
                                           [0.02027804, 0.01973165, 0.01722224, 0.01348521, 0.00947366],
                                           [0.01589901, 0.01546063, 0.01348521, 0.01055154, 0.00740717],
                                           [0.01118485, 0.01086915, 0.00947366, 0.00740717, 0.00519577]])
        assert np.allclose(sp1_convolved, sp1_convolved_expected, atol = 1e-5)
        assert np.allclose(sp1_convolved_default, sp1_convolved_expected, atol = 1e-5)
        
        # Test sparse_convolution (when the kernel size is smaller than the image size)
        conolved1 = psr_small_kernel.sparse_convolution([[0,0,0.8]])
        conolved2 = psr_small_kernel.sparse_convolution([[0,1,0.8]])
        conolved3 = psr_small_kernel.sparse_convolution([[1,1,0.8]])
        conolved4 = psr_small_kernel.sparse_convolution([[4,4,0.8]])
        
        compare1 = conolved1.copy()
        compare1[:2,:2] -= 0.8 * kernel_small[1:,1:]
        compare2 = conolved2.copy()
        compare2[:2,:3] -= 0.8 * kernel_small[1:]
        compare3 = conolved3.copy()
        compare3[:3,:3] -= 0.8 * kernel_small
        compare4 = conolved4.copy()
        compare4[3:,3:] -= 0.8 * kernel_small[:2,:2]
        
        assert np.allclose(compare1, 0, atol = 1e-5)
        assert np.allclose(compare2, 0, atol = 1e-5)
        assert np.allclose(compare3, 0, atol = 1e-5)
        assert np.allclose(compare4, 0, atol = 1e-5)
     
=======
        sp1_convolved_expected = np.array(
            [
                [0.02381713, 0.0232033, 0.02027804, 0.01589901, 0.01118485],
                [0.0232033, 0.02259202, 0.01973165, 0.01546063, 0.01086915],
                [0.02027804, 0.01973165, 0.01722224, 0.01348521, 0.00947366],
                [0.01589901, 0.01546063, 0.01348521, 0.01055154, 0.00740717],
                [0.01118485, 0.01086915, 0.00947366, 0.00740717, 0.00519577],
            ]
        )
        assert np.allclose(sp1_convolved, sp1_convolved_expected, atol=1e-5)


>>>>>>> 8389a9906b9db3877f2bc0296b72e90037f4489e
if __name__ == "__main__":
    pytest.main()
