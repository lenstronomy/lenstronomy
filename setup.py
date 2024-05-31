#!/usr/bin/env python

import os
import sys
from setuptools.command.test import test as TestCommand
from setuptools import find_packages

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
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


readme = open("README.rst").read()
doclink = """
Documentation
-------------

The full documentation can be generated with Sphinx"""

history = open("HISTORY.rst").read().replace(".. :changelog:", "")

desc = open("README.rst").read()
requires = [
    "numpy>=1.13",
    "scipy>=0.19.1",
    "configparser",
    "astropy",
    "mpmath",
    "matplotlib",
    "scikit-learn",
    "numba>=0.43.1",
    "corner>=2.2.1",
    "scikit-image",
    "pyyaml",
    "h5py",
    "pyxdg",
    "schwimmbad",
    "multiprocess>=0.70.8",
]
tests_require = [
    "pytest>=2.3",
    "mock",
    "colossus==1.3.0",
    "slitronomy==0.3.2",
    "emcee>=3.0.0",
    "dynesty",
    "nestcheck",
    "pymultinest",
    "zeus-mcmc>=2.4.0",
    "nautilus-sampler>=0.2.1",
    "coolest",
    "starred-astro>=1.4.2",
]

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))


setup(
    name="lenstronomy",
    version="1.11.10",
    description="Strong lens modeling package.",
    long_description=desc,
    author="lenstronomy developers",
    author_email="lenstronomy-dev@googlegroups.com",
    url="https://github.com/lenstronomy/lenstronomy",
    download_url="https://github.com/lenstronomy/lenstronomy/archive/1.11.10.tar.gz",
    packages=find_packages(PACKAGE_PATH, "test"),
    package_dir={"lenstronomy": "lenstronomy"},
    include_package_data=True,
    # setup_requires=requires,
    install_requires=requires,
    license="BSD-3",
    zip_safe=False,
    keywords="lenstronomy",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.11",
    ],
    tests_require=tests_require,
    cmdclass={"test": PyTest},  # 'build_ext':build_ext,
)
