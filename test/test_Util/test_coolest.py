import pytest
import numpy.testing as npt
import numpy as np
import unittest
import os

from lenstronomy.Util.coolest_interface import create_lenstro_from_coolest,update_coolest_from_lenstro

class TestCOOLESTinterface(object):

    def test_load(self):
        path = os.getcwd()
        kwargs_out = create_lenstro_from_coolest(path+"/coolest_template")
        print(kwargs_out)
        return