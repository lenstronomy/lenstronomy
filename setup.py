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

requires = ["sympy", "mpmath", 'numpy>=1.9.1', 'scipy>=0.14.0', "configparser", "astropy"] #during runtime ,, "darkskysync", "configparser", 'astropy', "PyCosmo", 'numpy>=1.7'
tests_require=['pytest>=2.3', "mock"] #for testing

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))


setup(
    name='lenstronomy',
    version='0.1.0',
    description='This package is designed to model strong lens systems.',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    author='Simon Birrer',
    author_email='simon.birrer@phys.ethz.ch',
    url='http://www.astro.ethz.ch/refregier/research/index',
    packages=find_packages(PACKAGE_PATH, "test"),
    package_dir={'lenstronomy': 'lenstronomy'},
    include_package_data=True,
    #setup_requires=requires,
    install_requires=requires,
    license='Proprietary',
    zip_safe=False,
    keywords='lenstronomy',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        "Intended Audience :: Science/Research",
        'Intended Audience :: Developers',
        'License :: Other/Proprietary License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
    ],
    tests_require=tests_require,
    cmdclass = { 'test': PyTest },#'build_ext':build_ext,
)
