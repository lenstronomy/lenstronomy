=============================
lenstronomy
=============================

This package is designed to model strong lens systems.
The software is based on Birrer et al 2015, http://adsabs.harvard.edu/abs/2015ApJ...813..102B and finds application in
e.g. Birrer et al. 2016 for time-delay cosmography and Birrer et al. 2017 for lensing substructure analysis.

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
* Modular build up to designe plugins by users
* Interactive jupyter notebooks
* Pre-defined plotting and ilustration routines
* Particle swarm optimization for parameter fitting
* MCMC (emcee from CosmoHammer)
* Cosmographic inference tools


