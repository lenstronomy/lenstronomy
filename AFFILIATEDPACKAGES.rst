===================
Affiliated packages
===================

Here is an (incomplete) list of packages and wrappers that are using lenstronomy in various ways for specific scientific
applications:

- `baobab <https://github.com/jiwoncpark/baobab>`_: Training data generator for hierarchically modeling of strong lenses with Bayesian neural networks.
- `dolphin <https://github.com/ajshajib/dolphin>`_: Automated pipeline for lens modeling based on lenstronomy.
- `hierArc <https://github.com/sibirrer/hierarc>`_: Hierarchical Bayesian time-delay cosmography to infer the Hubble constant and galaxy density profiles in conjunction with lenstronomy.
- `lenstruction <https://github.com/ylilan/lenstruction>`_: Versatile tool for cluster source reconstruction and local perturbative lens modeling.
- `SLITronomy <https://github.com/aymgal/SLITronomy>`_: Updated and improved version of the Sparse Lens Inversion Technique (SLIT), developed within the framework of lenstronomy.
- `LSSTDESC SLSprinkler <https://github.com/LSSTDESC/SLSprinkler>`_: The DESC SL (Strong Lensing) Sprinkler adds strongly lensed AGN and SNe to simulated catalogs and generates postage stamps for these systems.
- `lensingGW <https://gitlab.com/gpagano/lensinggw>`_: A Python package designed to handle both strong and microlensing of compact binaries and the related gravitational-wave signals.
- `ovejero <https://github.com/swagnercarena/ovejero>`_: Conducts hierarchical inference of strongly-lensed systems with Bayesian neural networks.
- `h0rton <https://github.com/jiwoncpark/h0rton>`_: H0 inferences with Bayesian neural network lens modeling.
- `deeplenstronomy <https://github.com/deepskies/deeplenstronomy>`_: Tool for simulating large datasets for applying deep learning to strong gravitational lensing.
- `pyHalo <https://github.com/dangilman/pyHalo>`_: Tool for rendering full substructure mass distributions for gravitational lensing simulations.
- `GaLight <https://github.com/dartoon/galight>`_: Tool to perform two-dimensional model fitting of optical and near-infrared images to characterize surface brightness distributions.
- `paltas <https://github.com/swagnercarena/paltas>`_: Package for conducting simulation-based inference on strong gravitational lensing images.
- `LensingETC <https://github.com/ajshajib/LensingETC>`_: A Python package to select an optimal observing strategy for multi-filter imaging campaigns of strong lensing systems. This package simulates imaging data corresponding to provided instrument specifications and extract lens model parameter uncertainties from the simulated images.
- `PSF-r <https://github.com/sibirrer/psfr>`_: Package for Point Spread Function (PSF) reconstruction for astronomical ground- and space-based imaging data. PSF-r makes use the PSF iteration functionality of lenstronomy in a re-packaged form.



These packages come with their own documentation and examples - so check them out!



Guidelines for affiliated packages
----------------------------------
If you have a package/wrapper/analysis pipeline that is open source and you would like to have it advertised here, please let the developers know!
Before you write your own wrapper and scripts in executing lenstronomy for your purpose check out the list
of existing add-on packages. Affiliated packages should not duplicate the core routines of lenstronomy and whenever possible make use of the lenstronomy modules.
The packages should be maintained to keep up with the development of lenstronomy. Please also make sure the citation guidelines are presented.