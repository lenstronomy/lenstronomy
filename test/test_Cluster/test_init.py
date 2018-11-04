__author__ = 'lilan'


import pytest
import lenstronomy.Cluster.hello as hello


class TestInit(object):

    def setup(self):
        pass

    def test_hello(self):
        a = 1
        b = 2
        c = hello.add1(a, b)
        assert c == 3


if __name__ == '__main__':
    pytest.main()