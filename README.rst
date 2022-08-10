====================================================
lenstronomy - gravitational lensing software package
====================================================


.. image:: https://github.com/lenstronomy/lenstronomy/workflows/Tests/badge.svg
    :target: https://github.com/lenstronomy/lenstronomy/actions

.. image:: https://readthedocs.org/projects/lenstronomy/badge/?version=latest
        :target: http://lenstronomy.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/lenstronomy/lenstronomy/badge.svg?branch=main
        :target: https://coveralls.io/github/lenstronomy/lenstronomy?branch=main

.. image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/lenstronomy/lenstronomy/blob/main/LICENSE

.. image:: https://img.shields.io/badge/arXiv-1803.09746%20-yellowgreen.svg
    :target: https://arxiv.org/abs/1803.09746

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
        :target: http://www.astropy.org
        :alt: Powered by Astropy Badge

.. image:: https://joss.theoj.org/papers/6a562375312c9a9e4466912a16f27589/status.svg
    :target: https://joss.theoj.org/papers/6a562375312c9a9e4466912a16f27589

.. image:: https://raw.githubusercontent.com/lenstronomy/lenstronomy/main/docs/figures/readme_fig.png
    :target: https://raw.githubusercontent.com/lenstronomy/lenstronomy/main/docs/figures/readme_fig.png


``lenstronomy`` is a multi-purpose package to model strong gravitational lenses. The software package is presented in
`Birrer & Amara 2018 <https://arxiv.org/abs/1803.09746v1>`_ and `Birrer et al. 2021 <https://joss.theoj.org/papers/10.21105/joss.03283>`_ , and is based on `Birrer et al 2015 <http://adsabs.harvard.edu/abs/2015ApJ...813..102B>`_.
``lenstronomy`` finds application for time-delay cosmography and measuring
the expansion rate of the Universe, for quantifying lensing substructure to infer dark matter properties, morphological quantification of galaxies,
quasar-host galaxy decomposition and much more.
A (incomplete) list of publications making use of lenstronomy can be found `at this link <https://github.com/lenstronomy/lenstronomy/blob/main/PUBLISHED.rst>`_.


The development is coordinated on `GitHub <https://github.com/lenstronomy/lenstronomy>`_ and contributions are welcome.
The documentation of ``lenstronomy`` is available at `readthedocs.org <http://lenstronomy.readthedocs.org/>`_ and
the package is distributed through PyPI_ and conda-forge_.
``lenstronomy`` is an `affiliated package <https://www.astropy.org/affiliated/>`_ of `astropy <https://www.astropy.org/>`_.



Installation
------------

|PyPI| |conda-forge|

lenstronomy releases are distributed through PyPI_ and conda-forge_. Instructions for
installing lenstronomy and its dependencies can be found in the Installation_
section of the documentation.
Specific instructions for settings and installation requirements for special cases that can provide speed-ups,
we also refer to the Installation_ page.

.. |PyPI| image:: https://img.shields.io/pypi/v/lenstronomy?label=PyPI&logo=pypi
    :target: https://pypi.python.org/pypi/lenstronomy

.. |conda-forge| image:: https://img.shields.io/conda/vn/conda-forge/lenstronomy?logo=conda-forge
    :target: https://anaconda.org/conda-forge/lenstronomy

.. _PyPI: https://pypi.org/project/lenstronomy/
.. _conda-forge: https://anaconda.org/conda-forge/lenstronomy
.. _Installation: https://lenstronomy.readthedocs.io/en/stable/installation.html


Getting started
---------------

The `starting guide jupyter notebook <https://github.com/lenstronomy/lenstronomy-tutorials/blob/main/Notebooks/GettingStarted/starting_guide.ipynb>`_
leads through the main modules and design features of ``lenstronomy``. The modular design of ``lenstronomy`` allows the
user to directly access a lot of tools and each module can also be used as stand-alone packages.

If you are new to gravitational lensing, check out the `mini lecture series <https://github.com/sibirrer/strong_lensing_lectures>`_ giving an introduction to gravitational lensing
with interactive Jupyter notebooks in the cloud.



Example notebooks
-----------------

We have made an extension module available at `https://github.com/lenstronomy/lenstronomy-tutorials <https://github.com/lenstronomy/lenstronomy-tutorials>`_.
You can find simple example notebooks for various cases. The latest versions of the notebooks should be compatible with the recent pip version of lenstronomy.



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
Please send us an `email <lenstronomy-dev@googlegroups.com>`_ such that we can add you to the channel.


If you encounter errors or problems with ``lenstronomy``, please let us know!



Contribution
------------
Check out the `contributing page <https://lenstronomy.readthedocs.io/en/latest/contributing.html>`_
and become an author of ``lenstronomy``! A big shout-out to the current `list of contributors and developers <https://lenstronomy.readthedocs.io/en/latest/authors.html>`_!



Attribution
-----------
The design concept of ``lenstronomy`` is reported by `Birrer & Amara 2018 <https://arxiv.org/abs/1803.09746v1>`_.
The current JOSS software publication is presented by `Birrer et al. 2021 <https://joss.theoj.org/papers/10.21105/joss.03283>`_.
Please cite these two publications when you use lenstronomy in a publication and link to `https://github.com/lenstronomy/lenstronomy <https://github.com/sibirrer/lenstronomy>`_.
Please also cite `Birrer et al 2015 <http://adsabs.harvard.edu/abs/2015ApJ...813..102B>`_
when you make use of the ``lenstronomy`` work-flow or the Shapelet source reconstruction and make sure to cite also
the relevant work that was implemented in ``lenstronomy``, as described in the release paper and the documentation.
Don't hesitate to reach out to the developers if you have questions!
