====================================================
lenstronomy - gravitational lensing software package
====================================================


.. image:: https://badge.fury.io/py/lenstronomy.png
    :target: https://badge.fury.io/py/lenstronomy

.. image:: https://travis-ci.org/sibirrer/lenstronomy.png?branch=main
        :target: https://travis-ci.org/sibirrer/lenstronomy

.. image:: https://readthedocs.org/projects/lenstronomy/badge/?version=latest
        :target: http://lenstronomy.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/sibirrer/lenstronomy/badge.svg?branch=main
        :target: https://coveralls.io/github/sibirrer/lenstronomy?branch=main

.. image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/sibirrer/lenstronomy/blob/main/LICENSE

.. image:: https://img.shields.io/badge/arXiv-1803.09746%20-yellowgreen.svg
    :target: https://arxiv.org/abs/1803.09746

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
        :target: http://www.astropy.org
        :alt: Powered by Astropy Badge

.. image:: https://joss.theoj.org/papers/6a562375312c9a9e4466912a16f27589/status.svg
    :target: https://joss.theoj.org/papers/6a562375312c9a9e4466912a16f27589

.. image:: https://raw.githubusercontent.com/sibirrer/lenstronomy/main/docs/figures/readme_fig.png
    :target: https://raw.githubusercontent.com/sibirrer/lenstronomy/main/docs/figures/readme_fig.png


``lenstronomy`` is a multi-purpose package to model strong gravitational lenses. The software package is presented in
`Birrer & Amara 2018 <https://arxiv.org/abs/1803.09746v1>`_ and is based on `Birrer et al 2015 <http://adsabs.harvard.edu/abs/2015ApJ...813..102B>`_.
``lenstronomy`` finds application for time-delay cosmography and measuring
the expansion rate of the universe, for quantifying lensing substructure to infer dark matter properties, morphological quantification of galaxies,
quasar-host galaxy decomposition and much more.
A (incomplete) list of publications making use of lenstronomy can be found `at this link <https://github.com/sibirrer/lenstronomy/blob/main/PUBLISHED.rst>`_.


The development is coordinated on `GitHub <https://github.com/sibirrer/lenstronomy>`_ and contributions are welcome.
The documentation of ``lenstronomy`` is available at `readthedocs.org <http://lenstronomy.readthedocs.org/>`_ and
the package is distributed over `PyPI <https://pypi.python.org/pypi/lenstronomy>`_.
``lenstronomy`` is an `affiliated package <https://www.astropy.org/affiliated/>`_ of `astropy <https://www.astropy.org/>`_.



Installation
------------

.. code-block:: bash

    $ pip install lenstronomy --user


Specific instructions for settings and installation requirements for special cases that can provide speed-ups,
we refer to the `documentation <https://lenstronomy.readthedocs.io/en/latest/installation.html>`_ page.



Getting started
---------------

The `starting guide jupyter notebook <https://github.com/sibirrer/lenstronomy_extensions/blob/main/lenstronomy_extensions/Notebooks/starting_guide.ipynb>`_
leads through the main modules and design features of ``lenstronomy``. The modular design of ``lenstronomy`` allows the
user to directly access a lot of tools and each module can also be used as stand-alone packages.


Example notebooks
-----------------

We have made an extension module available at `https://github.com/sibirrer/lenstronomy_extensions <https://github.com/sibirrer/lenstronomy_extensions>`_.
You can find simple examle notebooks for various cases. The latest versions of the notebooks should be compatible with the recent pip version of lenstronomy.

* `Units, coordinate system and parameter definitions in lenstronomy <https://github.com/sibirrer/lenstronomy_extensions/blob/main/lenstronomy_extensions/Notebooks/units_coordinates_parameters.ipynb>`_
* `FITS handling and extracting needed information from the data prior to modeling <https://github.com/sibirrer/lenstronomy_extensions/blob/main/lenstronomy_extensions/Notebooks/fits_handling.ipynb>`_
* `Modeling a simple Einstein ring <https://github.com/sibirrer/lenstronomy_extensions/blob/main/lenstronomy_extensions/Notebooks/simple_ring.ipynb>`_
* `Quadrupoly lensed quasar modelling <https://github.com/sibirrer/lenstronomy_extensions/blob/main/lenstronomy_extensions/Notebooks/quad_model.ipynb>`_
* `Double lensed quasar modelling <https://github.com/sibirrer/lenstronomy_extensions/blob/main/lenstronomy_extensions/Notebooks/double_model.ipynb>`_
* `Time-delay cosmography <https://github.com/sibirrer/lenstronomy_extensions/blob/main/lenstronomy_extensions/Notebooks/time-delay%20cosmography.ipynb>`_
* `Source reconstruction and deconvolution with Shapelets <https://github.com/sibirrer/lenstronomy_extensions/blob/main/lenstronomy_extensions/Notebooks/shapelet_source_modelling.ipynb>`_
* `Solving the lens equation <https://github.com/sibirrer/lenstronomy_extensions/blob/main/lenstronomy_extensions/Notebooks/lens_equation.ipynb>`_
* `Multi-band fitting <https://github.com/sibirrer/lenstronomy_extensions/blob/main/lenstronomy_extensions/Notebooks/multi_band_fitting.ipynb>`_
* `Measuring cosmic shear with Einstein rings <https://github.com/sibirrer/lenstronomy_extensions/blob/main/lenstronomy_extensions/Notebooks/EinsteinRingShear_simulations.ipynb>`_
* `Fitting of galaxy light profiles, like e.g. GALFIT <https://github.com/sibirrer/lenstronomy_extensions/blob/main/lenstronomy_extensions/Notebooks/galfitting.ipynb>`_
* `Quasar-host galaxy decomposition <https://github.com/sibirrer/lenstronomy_extensions/blob/main/lenstronomy_extensions/Notebooks/quasar-host%20decomposition.ipynb>`_
* `Hiding and seeking a single subclump <https://github.com/sibirrer/lenstronomy_extensions/blob/main/lenstronomy_extensions/Notebooks/substructure_challenge_simple.ipynb>`_
* `Mock generation of realistic images with substructure in the lens <https://github.com/sibirrer/lenstronomy_extensions/blob/main/lenstronomy_extensions/Notebooks/substructure_challenge_mock_production.ipynb>`_
* `Mock simulation API with multi color models <https://github.com/sibirrer/lenstronomy_extensions/blob/main/lenstronomy_extensions/Notebooks/simulation_api.ipynb>`_
* `Catalogue data modeling of image positions, flux ratios and time delays <https://github.com/sibirrer/lenstronomy_extensions/blob/main/lenstronomy_extensions/Notebooks/catalogue%20modelling.ipynb>`_
* `Example of numerical ray-tracing and convolution options <https://github.com/sibirrer/lenstronomy_extensions/blob/main/lenstronomy_extensions/Notebooks/lenstronomy_numerics.ipynb>`_
* `Simulated lenses with populations generated by SkyPy <https://github.com/sibirrer/lenstronomy_extensions/blob/main/lenstronomy_extensions/Notebooks/skypy_lenstronomy.ipynb>`_



Affiliated packages
-------------------
Multiple affiliated packages that make use of lenstronomy can be found `here <https://lenstronomy.readthedocs.io/en/latest/affiliatedpackages.html>`_
(not complete) and further packages are under development by the community.


Mailing list and Slack channel
------------------------------

You can join the ``lenstronomy`` mailing list by signing up on the
`google groups page <https://groups.google.com/forum/#!forum/lenstronomy>`_.


The email list is meant to provide a communication platform between users and developers. You can ask questions,
and suggest new features. New releases will be announced via this mailing list.

We also have a `Slack channel <https://lenstronomers.slack.com>`_ for the community.
Please send me an `email <sibirrer@gmail.com>`_ such that I can add you to the channel.


If you encounter errors or problems with ``lenstronomy``, please let us know!



Contribution
------------
Check out the `contributing page <https://lenstronomy.readthedocs.io/en/latest/contributing.html>`_
and become an author of ``lenstronomy``! A big shutout to the current `list of contributors and developers <https://lenstronomy.readthedocs.io/en/latest/authors.html>`_!




Shapelet reconstruction demonstration movies
--------------------------------------------

We provide some examples where a real galaxy has been lensed and then been reconstructed by a shapelet basis set.

* `HST quality data with perfect knowledge of the lens model <http://www.astro.ucla.edu/~sibirrer/video/true_reconstruct.mp4>`_
* `HST quality with a clump hidden in the data <http://www.astro.ucla.edu/~sibirrer/video/clump_reconstruct.mp4>`_
* `Extremely large telescope quality data with a clump hidden in the data <http://www.astro.ucla.edu/~sibirrer/video/TMT_high_res_clump_reconstruct.mp4>`_



Attribution
-----------
The design concept of ``lenstronomy`` is reported by `Birrer & Amara 2018 <https://arxiv.org/abs/1803.09746v1>`_.
The current JOSS software publication is presented by `Birrer et al. 2021 <https://joss.theoj.org/papers/10.21105/joss.03283>`_.
Please cite these two publications when you use lenstronomy in a publication and link to `https://github.com/sibirrer/lenstronomy <https://github.com/sibirrer/lenstronomy>`_.
Please also cite `Birrer et al 2015 <http://adsabs.harvard.edu/abs/2015ApJ...813..102B>`_
when you make use of the ``lenstronomy`` work-flow or the Shapelet source reconstruction and make sure to cite also
the relevant work that was implemented in ``lenstronomy``, as described in the release paper and the documentation.
Don't hesitate to reach out to the developers if you have questions!
