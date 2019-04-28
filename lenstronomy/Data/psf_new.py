

class PSF(object):
    """
    class to manage the point-spread-function attributes
    """
    def __init__(self):
        pass

    @property
    def pixel_kernel(self):
        """

        :return: 2d gird of PSF kernel in pixel units
        """
        return 0

    def subsampled_kernel(self, subsampling_factor):
        """

        :param subsampling_factor: int, subsampling factor
        :return: subsampled PSF
        """
        return 0