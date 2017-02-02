__author__ = 'sibirrer'


import numpy as np

class InitialiseFromChain(object):
    """

        Picks random positions from a given sample to initialize the walkers.
    """

    def __init__(self, path, fraction = 2):
        """
            default constructor
        """
        self.path = path
        self.fraction = fraction

    def setup(self, sampler):
        sample = np.loadtxt(self.path)
        if sample.shape[1] != sampler.paramCount:
            raise Warning('Sample dimensions do not agree with likelihood ones.')
        self.n = int(sample.shape[0]/float(self.fraction))
        self.sample = sample[self.n:]
        self.nwalkers = sampler.nwalkers

    def generate(self):
        """
            generates the positions
        """
        pos = np.random.randint(0, self.n, size=self.nwalkers)

        return self.sample[pos]

    def __str__(self, *args, **kwargs):
        return "InitialiseFromChain"