=============================
lenstronomy - the gravitational lensing software package
=============================

.. image:: https://badge.fury.io/py/lenstronomy.png
    :target: http://badge.fury.io/py/lenstronomy

.. image:: https://travis-ci.org/sibirrer/lenstronomy.png?branch=master
        :target: https://travis-ci.org/sibirrer/lenstronomy

.. image:: https://readthedocs.org/projects/lenstronomy/badge/?version=latest
        :target: http://lenstronomy.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/sibirrer/lenstronomy/badge.svg?branch=master
        :target: https://coveralls.io/github/sibirrer/lenstronomy?branch=master

.. image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/sibirrer/lenstronomy/blob/master/LICENSE


The model package for gravitational strong lens images.
The software is based on `Birrer et al 2015 <http://adsabs.harvard.edu/abs/2015ApJ...813..102B>`_  and finds application in
e.g. Birrer et al. 2016 for time-delay cosmography and Birrer et al. 2017 for lensing substructure analysis.


The development is coordinated on `GitHub <https://github.com/sibirrer/lenstronomy>`_ and contributions are welcome.
The documentation of **lenstronomy** is available at `readthedocs.org <http://lenstronomy.readthedocs.org/>`_ and
the package is distributed over `PyPI <https://pypi.python.org/pypi/lenstronomy>`_.



Installation
--------

.. code-block:: bash

    $ pip install lenstronomy


Requirements
-------
To run lens models with elliptical mass distributions, the fastell4py package, originally from Barkana (fastell),
is also required and can be cloned from: `https://github.com/sibirrer/fastell4py <https://github.com/sibirrer/fastell4py>`_ (needs a fortran compiler)
* CosmoHammer (through PyPi)
* standard python libraries (numpy, scipy)


Bug reporting and contributions
-------
* see CONTRIBUTING.rst


Modelling Features
--------

* Extended source reconstruction with basis sets (shapelets)
* Analytic light profiles for lens and source as options
* Point sources (including solving the lens equation)
* a variety of mass models to use
* non-linear line-of-sight description
* iterative point spread function
* linear and non-linear optimization modules



Analysis tools
-------
* Standardized fitting procedures for lens modelling
* Modular build up to design plugins by users
* Pre-defined plotting and illustration routines
* Particle swarm optimization for parameter fitting
* MCMC (emcee from CosmoHammer) for parameter inferences
* Kinematic modelling
* Cosmographic inference tools



Example notebooks
------
We have made an extension module available at `http://github.com/sibirrer/lenstronomy_extensions <http://github.com/sibirrer/lenstronomy_extensions>`_ .
You can find examle notebooks for various cases, such as time-delay cosmography, substructure lensing,
line-of-sight analysis and source reconstructions.
