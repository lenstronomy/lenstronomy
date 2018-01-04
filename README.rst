=============================
lenstronomy
=============================

.. image:: https://travis-ci.org/sibirrer/lenstronomy.png?branch=master
        :target: https://travis-ci.org/sibirrer/lenstronomy

.. image:: https://readthedocs.org/projects/lenstronomy/badge/?version=latest
        :target: http://lenstronomy.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/sibirrer/lenstronomy/badge.svg?branch=master
        :target: https://coveralls.io/github/sibirrer/lenstronomy?branch=master


This package is designed to model strong lens systems.
The software is based on Birrer et al 2015, http://adsabs.harvard.edu/abs/2015ApJ...813..102B and finds application in
e.g. Birrer et al. 2016 for time-delay cosmography and Birrer et al. 2017 for lensing substructure analysis.


The development is coordinated on `GitHub <http://github.com/sibirrer/lenstronomy>`_ and contributions are welcome.
The documentation of **lenstronomy** is available at `readthedocs.org <http://lenstronomy.readthedocs.org/>`_



Installation
--------
* check out the github repository
>>> cd lenstronomy
>>> python setup.py install
or in development mode
>>> python setup.py develop
* it is recommended to check out and install the dependency fastell4py independently.
* run the test functions to see whether the installation was successful.
>>> cd lenstronomy
>>> py.test


Requirements
-------
* to run the lens models from Barkana (fastell), also requires githug/sibirrer/fastell4py (needs a fortran compiler)
* CosmoHammer (through PyPi)
* standard python libraries (numpy, scipy)


Bug reporting and contributions
-------
* see CONTRIBUTING.rst
* you can also directly contact the lead developer, Simon Birrer

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
* Standardized fitting procedures for some lens systems
* Modular build up to design plugins by users
* Interactive jupyter notebooks
* Pre-defined plotting and illustration routines
* Particle swarm optimization for parameter fitting
* MCMC (emcee from CosmoHammer) for parameter inferences
* Kinematic modelling
* Cosmographic inference tools


