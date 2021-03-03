============
Installation
============

At the command line with pip::

    $ pip install lenstronomy

Or, if you have virtualenvwrapper installed::

    $ mkvirtualenv lenstronomy
    $ pip install lenstronomy

You can also clone the github repository for development purposes.


Requirements
------------

Make sure the standard python libraries as specified in the `requirements <https://github.com/sibirrer/lenstronomy/blob/master/requirements.txt>`_.
The standard usage does not require all libraries to be installed, in particluar the different posterior samplers are only required when being used.

In the following, a few specific cases are mentioned that may require special attention in the installation and settings, in particular when it comes
to MPI and HPC applications.


MPI
---
MPI support is provided for several sampling techniques for parallel computing. A specific version of the library schwimmbad is required
for the correct support of the moving of the likelihood elements from one processor to another with MPI. Pay attention ot the
`requirements <https://github.com/sibirrer/lenstronomy/blob/master/requirements.txt>`_.


NUMBA
-----
Just-in-time (jit) compilation with numba can provide significant speed-up for certain calculations.
There are specific settings for the settings provided as per default, but these may need to be adopted when running on a HPC cluster.


FASTELL
-------
The fastell4py package, originally from Barkana (fastell), is required to run the PEMD (power-law elliptical mass distribution) lens model
and can be cloned from: `https://github.com/sibirrer/fastell4py <https://github.com/sibirrer/fastell4py>`_ (needs a fortran compiler).
We recommend using the EPL model as it is a pure python version of the same profile.

.. code-block:: bash

    $ sudo apt-get install gfortran
    $ git clone https://github.com/sibirrer/fastell4py.git <desired location>
    $ cd <desired location>
    $ python setup.py install --user
