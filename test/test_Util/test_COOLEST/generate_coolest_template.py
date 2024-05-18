# Script that generates a a baseline JSON template following the COOLEST standard.
# The template is then used by the test routines in test/test_Util/test_COOLEST in order to cover the unit tests.
# Note the name of the file should *not* contain "test" to be ignored by pytest.


import os
import numpy as np
from astropy.io import fits
from coolest.template.lazy import *
from coolest.template.standard import COOLEST
from coolest.template.json import JSONSerializer

# from coolest.template.classes.probabilities import GaussianPrior
# from coolest.template.classes.parameter import PointEstimate
# from coolest.template.classes.probabilities import PosteriorStatistics


# basename of the generated COOLEST JSON template (no extension given here)
TEMPLATE_NAME = "coolest_template"

# purposedly introduce COOLEST options that are not yet supported by the lenstronomy's COOLEST interface
WITH_UNSUPPORTED_OPTION = True
UNSUPPORTED_OPTION_TYPE = "lenslight"  # "noise", "psf", "multiplane", "sourcelight", "lensmass", "lenslight", "massfield"


# Setup cosmology
cosmology = Cosmology(H0=73.0, Om0=0.3)

# Create a source galaxy represented by light profile
if WITH_UNSUPPORTED_OPTION and UNSUPPORTED_OPTION_TYPE == "sourcelight":
    main_source = Galaxy(
        "a source galaxy", 2.0, light_model=LightModel("PixelatedRegularGrid")
    )
else:
    main_source = Galaxy(
        "a source galaxy",
        2.0,
        light_model=LightModel("Sersic", "Shapelets", "LensedPS"),
    )

# Create a lens galaxy represented by a mass and light profiles
if WITH_UNSUPPORTED_OPTION and UNSUPPORTED_OPTION_TYPE == "lensmass":
    main_lens = Galaxy(
        "a lens galaxy",
        0.5,
        light_model=LightModel("Sersic"),
        mass_model=MassModel("PixelatedRegularGridPotential"),
    )
elif WITH_UNSUPPORTED_OPTION and UNSUPPORTED_OPTION_TYPE == "lenslight":
    main_lens = Galaxy(
        "a lens galaxy",
        0.5,
        light_model=LightModel("PixelatedRegularGrid"),
        mass_model=MassModel("SIE", "PEMD"),
    )
else:
    main_lens = Galaxy(
        "a lens galaxy",
        0.5,
        light_model=LightModel("Sersic"),
        mass_model=MassModel("SIE", "PEMD"),
    )

# Defines the external shear as a massive field
ext_shear = MassField(
    "an external shear",
    main_lens.redshift,
    mass_model=MassModel("ExternalShear", "ConvergenceSheet"),
)

# Put them in a list, which will also create unique IDs for each profile
if WITH_UNSUPPORTED_OPTION and UNSUPPORTED_OPTION_TYPE == "multiplane":
    lens_other_z = Galaxy(
        "a lens galaxy at different redshift",
        0.2,
        light_model=LightModel("Sersic"),
        mass_model=MassModel("SIE"),
    )
    source_other_z = Galaxy(
        "a source galaxy at different redshift", 1.3, light_model=LightModel("Sersic")
    )
    entity_list = LensingEntityList(
        ext_shear, main_lens, main_source, lens_other_z, source_other_z
    )
else:
    entity_list = LensingEntityList(ext_shear, main_lens, main_source)

# Define the origin of the coordinates system
origin = CoordinatesOrigin("00h11m20.244s", "-08d45m51.48s")

# EXAMPLE for accessing specific parameters and add priors/values/posteriors
# - add a gaussian prior to a given parameter
# main_lens.mass_model[0].parameters['gamma'].set_prior(GaussianPrior(mean=2.0, width=0.2))

# - add a point estimate to a given parameter
# ext_shear.mass_model[0].parameters['gamma_ext'].set_point_estimate(0.07)
# main_lens.light_model[1].parameters['q'].set_point_estimate(PointEstimate(value=0.89))

# - add a posterior distribution (as 0th and 1st order statistics)
# main_source.light_model[0].parameters['theta_eff'].set_posterior(PosteriorStatistics(mean=0.11, median=0.15, percentile_16th=0.03, percentile_84th=0.05))

# Define the data pixels and noise properties
data_shape = (100, 100)
obs_pixels = PixelatedRegularGrid(
    "obs.fits",
    field_of_view_x=[-4.0, 4.0],
    field_of_view_y=[-4.0, 4.0],
    num_pix_x=data_shape[0],
    num_pix_y=data_shape[1],
)
if WITH_UNSUPPORTED_OPTION and UNSUPPORTED_OPTION_TYPE == "noise":
    noise = UniformGaussianNoise(
        std_dev=0.005
    )  # other types of noise are supported in COOLEST
else:
    obs_noise_pixels = PixelatedRegularGrid(
        "noise_map.fits",
        field_of_view_x=[-4.0, 4.0],
        field_of_view_y=[-4.0, 4.0],
        num_pix_x=data_shape[0],
        num_pix_y=data_shape[1],
    )
    noise = NoiseMap(obs_noise_pixels)
observation = Observation(
    pixels=obs_pixels,
    noise=noise,  # other types of noise are supported in COOLEST
)

# Defines the instrument
if WITH_UNSUPPORTED_OPTION and UNSUPPORTED_OPTION_TYPE == "psf":
    psf = GaussianPSF(fwhm=0.8)
else:
    psf_shape = (23, 23)
    psf_pixels = PixelatedRegularGrid(
        "psf_kernel.fits",
        field_of_view_x=[-0.92, 0.92],
        field_of_view_y=[-0.92, 0.92],
        num_pix_x=psf_shape[0],
        num_pix_y=psf_shape[1],
    )
    psf = PixelatedPSF(psf_pixels)
instrument = Instrument(
    0.08,  # pixel size
    name="some instrument",
    readout_noise=4,
    band="F160W",
    psf=psf,  # other types of PSF are supported in COOLEST
)

# Master object for the standard
coolest = COOLEST("MAP", origin, entity_list, observation, instrument, cosmology)

# export as JSON file
output_filename = TEMPLATE_NAME
if WITH_UNSUPPORTED_OPTION:
    output_filename = "unsupp_" + output_filename + f"_{UNSUPPORTED_OPTION_TYPE}"
template_path = os.path.join(os.getcwd(), output_filename)
serializer = JSONSerializer(template_path, obj=coolest, check_external_files=True)
serializer.dump_simple()


# below we also create random fake FITS files that matches the above components
np.random.seed(0)
fits.writeto("obs.fits", np.random.randn(*data_shape), overwrite=True)
if not WITH_UNSUPPORTED_OPTION:
    fits.writeto("noise_map.fits", np.random.randn(*data_shape), overwrite=True)
    fits.writeto("psf_kernel.fits", np.random.randn(*psf_shape), overwrite=True)
