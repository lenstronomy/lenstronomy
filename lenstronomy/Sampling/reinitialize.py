"""
Created on May 18, 2015

author: jakeret
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np


class ReusePositionGenerator(object):
    
    def __init__(self, positions):
        self.positions = positions

    def setup(self, sampler):
        self.sampler = sampler
    
    def generate(self):
        nwalker = self.sampler.nwalkers
        idxs = np.random.choice(len(self.positions), nwalker)
        return self.positions[idxs]
    
    def __str__(self, *args, **kwargs):
        return "ReusePositionGenerator"