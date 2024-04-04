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


# TEMPLATE_NAME = "coolest_template_pemd"
TEMPLATE_NAME = "coolest_template"

# Setup cosmology
cosmology = Cosmology(H0=73.0, Om0=0.3)

# Create a source galaxy represented by light profile
source_1 = Galaxy("a source galaxy", 2.0, light_model=LightModel("Sersic", "Shapelets"))

# Create a lens galaxy represented by a mass and light profiles
lens_1 = Galaxy(
    "a lens galaxy",
    0.5,
    light_model=LightModel("Sersic"),
    mass_model=MassModel("SIE", "PEMD"),
)

# Defines the external shear as a massive field
ext_shear = MassField(
    "an external shear",
    lens_1.redshift,
    mass_model=MassModel("ExternalShear", "ConvergenceSheet"),
)

# Put them in a list, which will also create unique IDs for each profile
entity_list = LensingEntityList(ext_shear, lens_1, source_1)

# Define the origin of the coordinates system
origin = CoordinatesOrigin("00h11m20.244s", "-08d45m51.48s")

# EXAMPLE for accessing specific parameters and add priors/values/posteriors
# - add a gaussian prior to a given parameter
# lens_1.mass_model[0].parameters['gamma'].set_prior(GaussianPrior(mean=2.0, width=0.2))

# - add a point estimate to a given parameter
# ext_shear.mass_model[0].parameters['gamma_ext'].set_point_estimate(0.07)
# lens_1.light_model[1].parameters['q'].set_point_estimate(PointEstimate(value=0.89))

# - add a posterior distribution (as 0th and 1st order statistics)
# source_1.light_model[0].parameters['theta_eff'].set_posterior(PosteriorStatistics(mean=0.11, median=0.15, percentile_16th=0.03, percentile_84th=0.05))

# Define the data pixels and noise properties
data_shape = (100, 100)
obs_pixels = PixelatedRegularGrid(
    "obs.fits",
    field_of_view_x=[-4.0, 4.0],
    field_of_view_y=[-4.0, 4.0],
    num_pix_x=data_shape[0],
    num_pix_y=data_shape[1],
)
obs_noise_pixels = PixelatedRegularGrid(
    "noise_map.fits",
    field_of_view_x=[-4.0, 4.0],
    field_of_view_y=[-4.0, 4.0],
    num_pix_x=data_shape[0],
    num_pix_y=data_shape[1],
)
observation = Observation(
    pixels=obs_pixels,
    noise=NoiseMap(obs_noise_pixels),  # other types of noise are supported in COOLEST
)

# Defines the instrument
psf_shape = (23, 23)
psf_pixels = PixelatedRegularGrid(
    "psf_kernel.fits",
    field_of_view_x=[-0.92, 0.92],
    field_of_view_y=[-0.92, 0.92],
    num_pix_x=psf_shape[0],
    num_pix_y=psf_shape[1],
)
instrument = Instrument(
    0.08,  # pixel size
    name="some instrument",
    readout_noise=4,
    band="F160W",
    psf=PixelatedPSF(psf_pixels),  # other types of PSF are supported in COOLEST
)

# Master object for the standard
coolest = COOLEST("MAP", origin, entity_list, observation, instrument, cosmology)

# export as JSON file
template_path = os.path.join(os.getcwd(), TEMPLATE_NAME)
serializer = JSONSerializer(template_path, obj=coolest, check_external_files=True)
serializer.dump_simple()



# below we also create random fake FITS files that matches the above components
np.random.seed(0)
fits.writeto("obs.fits", np.random.randn(*data_shape), overwrite=True)
fits.writeto("noise_map.fits", np.random.randn(*data_shape), overwrite=True)
fits.writeto("psf_kernel.fits", np.random.randn(*psf_shape), overwrite=True)
