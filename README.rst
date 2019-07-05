========================================================
lenstronomy - gravitational lensing software package
========================================================

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

.. image:: https://img.shields.io/badge/arXiv-1803.09746%20-yellowgreen.svg
    :target: https://arxiv.org/abs/1803.09746

``lenstronomy`` is a multi-purpose package to model strong gravitational lenses. The software package is presented in
`Birrer & Amara 2018 <https://arxiv.org/abs/1803.09746v1>`_ and is based on `Birrer et al 2015 <http://adsabs.harvard.edu/abs/2015ApJ...813..102B>`_.
``lenstronomy`` finds application in e.g. `Birrer et al 2016 <http://adsabs.harvard.edu/abs/2016JCAP...08..020B>`_ and
`Birrer et al 2018 <http://adsabs.harvard.edu/abs/2018arXiv180901274B>`_ for time-delay cosmography and measuring
the expansion rate of the universe and `Birrer et al 2017 <http://adsabs.harvard.edu/abs/2017JCAP...05..037B>`_ for
quantifying lensing substructure to infer dark matter properties.


The development is coordinated on `GitHub <https://github.com/sibirrer/lenstronomy>`_ and contributions are welcome.
The documentation of ``lenstronomy`` is available at `readthedocs.org <http://lenstronomy.readthedocs.org/>`_ and
the package is distributed over `PyPI <https://pypi.python.org/pypi/lenstronomy>`_.



Installation
------------

.. code-block:: bash

    $ pip install lenstronomy --user


Requirements
------------
To run lens models with elliptical mass distributions, the fastell4py package, originally from Barkana (fastell),
is also required and can be cloned from: `https://github.com/sibirrer/fastell4py <https://github.com/sibirrer/fastell4py>`_ (needs a fortran compiler)

Additional python libraries:

* ``CosmoHammer`` (through PyPi)
* ``astropy``
* ``dynesty``
* ``pymultinest``
* ``pypolychord``
* ``nestcheck``
* standard python libraries (``numpy``, ``scipy``)



Modelling Features
------------------

* a variety of lens models to use in arbitrary superposition
* lens equation solver
* multi-plane ray-tracing
* Extended source reconstruction with basis sets (shapelets) and analytic light profiles
* Point sources
* numerical options for sub-grid ray-tracing and sub-pixel convolution
* non-linear line-of-sight description
* iterative point spread function reconstruction
* linear and non-linear optimization modules
* Pre-defined plotting and illustration routines
* Particle swarm optimization for parameter fitting
* MCMC (emcee from CosmoHammer) for parameter inferences
* Nested Sampling (MultiNest, DyPolyChord, or Dynesty) for evidence computation and parameter inferences
* Kinematic modelling
* Cosmographic inference tools



Getting started
---------------

The `starting guide jupyter notebook <https://github.com/sibirrer/lenstronomy_extensions/blob/master/lenstronomy_extensions/Notebooks/starting_guide.ipynb>`_
leads through the main modules and design features of ``lenstronomy``. The modular design of ``lenstronomy`` allows the
user to directly access a lot of tools and each module can also be used as stand-alone packages.


Example notebooks
-----------------

We have made an extension module available at `http://github.com/sibirrer/lenstronomy_extensions <https://github.com/sibirrer/lenstronomy_extensions>`_.
You can find simple examle notebooks for various cases.

* `Quadrupoly lensed quasar modelling <https://github.com/sibirrer/lenstronomy_extensions/blob/master/lenstronomy_extensions/Notebooks/quad_model.ipynb>`_
* `Double lensed quasar modelling <https://github.com/sibirrer/lenstronomy_extensions/blob/master/lenstronomy_extensions/Notebooks/double_model.ipynb>`_
* `Time-delay cosmography <https://github.com/sibirrer/lenstronomy_extensions/blob/master/lenstronomy_extensions/Notebooks/time-delay%20cosmography.ipynb>`_
* `Source reconstruction and deconvolution with Shapelets <https://github.com/sibirrer/lenstronomy_extensions/blob/master/lenstronomy_extensions/Notebooks/shapelet_source_modelling.ipynb>`_
* `Solving the lens equation <https://github.com/sibirrer/lenstronomy_extensions/blob/master/lenstronomy_extensions/Notebooks/lens_equation.ipynb>`_
* `Measuring cosmic shear with Einstein rings <https://github.com/sibirrer/lenstronomy_extensions/blob/master/lenstronomy_extensions/Notebooks/EinsteinRingShear_simulations.ipynb>`_
* `Fitting of galaxy light profiles, like e.g. GALFIT <https://github.com/sibirrer/lenstronomy_extensions/blob/master/lenstronomy_extensions/Notebooks/galfitting.ipynb>`_
* `Quasar-host galaxy decomposition <https://github.com/sibirrer/lenstronomy_extensions/blob/master/lenstronomy_extensions/Notebooks/quasar-host%20decomposition.ipynb>`_
* `Hiding and seeking a single subclump <https://github.com/sibirrer/lenstronomy_extensions/blob/master/lenstronomy_extensions/Notebooks/substructure_challenge_simple.ipynb>`_
* `Mock generation of realistic images with substructure in the lens <https://github.com/sibirrer/lenstronomy_extensions/blob/master/lenstronomy_extensions/Notebooks/substructure_challenge_mock_production.ipynb>`_
* `Mock simulation API with multi color models <https://github.com/sibirrer/lenstronomy_extensions/blob/master/lenstronomy_extensions/Notebooks/simulation_api.ipynb>`_
* `Catalogue data modeling of image positions, flux ratios and time delays <https://github.com/sibirrer/lenstronomy_extensions/blob/master/lenstronomy_extensions/Notebooks/catalogue%20modelling.ipynb>`_
* `Example of numerical ray-tracing and convolution options <https://github.com/sibirrer/lenstronomy_extensions/blob/master/lenstronomy_extensions/Notebooks/lenstronomy_numerics.ipynb>`_


Mailing list
------------

You can join the **lenstronomy** mailing list by signing up on the
`google groups page <https://groups.google.com/forum/#!forum/lenstronomy>`_.

The email list is meant to provide a communication platform between users and developers. You can ask questions,
and suggest new features. New releases will be announced via this mailing list.

If you encounter errors or problems with **lenstronomy**, please let us know!


Shapelet reconstruction demonstration movies
--------------------------------------------

We provide some examples where a real galaxy has been lensed and then been reconstructed by a shapelet basis set.

* `HST quality data with perfect knowledge of the lens model <http://www.astro.ucla.edu/~sibirrer/video/true_reconstruct.mp4>`_
* `HST quality with a clump hidden in the data <http://www.astro.ucla.edu/~sibirrer/video/clump_reconstruct.mp4>`_
* `Extremely large telescope quality data with a clump hidden in the data <http://www.astro.ucla.edu/~sibirrer/video/TMT_high_res_clump_reconstruct.mp4>`_



Attribution
-----------
The design concept of ``lenstronomy`` are reported in
`Birrer & Amara 2018 <https://arxiv.org/abs/1803.09746v1>`_. Please cite this paper whenever you publish
results that made use of ``lenstronomy``. Please also cite `Birrer et al 2015 <http://adsabs.harvard.edu/abs/2015ApJ...813..102B>`_
when you make use of the ``lenstronomy`` work-flow or the Shapelet source reconstruction. Please make sure to cite also
the relevant work that was implemented in ``lenstronomy``, as described in the release paper.
