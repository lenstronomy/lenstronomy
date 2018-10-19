#!/usr/bin/env python

import os
import sys
from setuptools.command.test import test as TestCommand
from setuptools import find_packages
from setuptools.command.build_ext import build_ext as _build_ext

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


readme = open('README.rst').read()
doclink = """
Documentation
-------------

The full documentation can be generated with Sphinx"""

history = open('HISTORY.rst').read().replace('.. :changelog:', '')

desc = open("README.rst").read()
requires = ['numpy>=1.13', 'scipy>=0.14.0', "configparser"]
tests_require=['pytest>=2.3', "mock"]

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))


setup(
    name='lenstronomy',
    version='0.4.1',
    description='Strong lens modeling package.',
    long_description=desc,
    author='Simon Birrer',
    author_email='sibirrer@gmail.com',
    url='https://github.com/sibirrer/lenstronomy',
    download_url='https://github.com/sibirrer/lenstronomy/archive/0.4.1.tar.gz',
    packages=find_packages(PACKAGE_PATH, "test"),
    package_dir={'lenstronomy': 'lenstronomy'},
    include_package_data=True,
    #setup_requires=requires,
    install_requires=requires,
    license='MIT',
    zip_safe=False,
    keywords='lenstronomy',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.6",
    ],
    tests_require=tests_require,
    cmdclass={'test': PyTest},#'build_ext':build_ext,
)
