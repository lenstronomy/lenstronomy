import numpy as np
import numpy.testing as npt
from lenstronomy.Data.imaging_data import ImageData


def test_interferometry_likelihood():
    """

    test interferometry natural weighting likelihood function, test likelihood_method function output

    """

    test_data = np.zeros((5, 5))
    test_data[0, :] = 1
    test_data[:, 4] = 2

    mask = np.ones((5, 5))
    model_unconvolved = np.zeros((5, 5))
    model_convolved = np.zeros((5, 5))
    model_unconvolved[0, 4] = 1.2
    model_convolved[0, :] = 1
    model_convolved[:, 4] = 1

    data_class = ImageData(
        image_data=test_data,
        background_rms=2.5,
        log_likelihood_constant=-1.0,
        likelihood_method="interferometry_natwt",
    )

    assert data_class.likelihood_method() == "interferometry_natwt"
    npt.assert_almost_equal(
        data_class.log_likelihood([model_unconvolved, model_convolved], mask),
        -0.712,
        decimal=8,
    )
