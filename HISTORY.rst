.. :changelog:

History
-------

0.0.1 (2018-01-09)
++++++++++++++++++

* First release on PyPI.

0.0.2 (2018-01-16)
++++++++++++++++++

* Improved testing and stability

0.0.6 (2018-01-29)
++++++++++++++++++

* Added feature to align coordinate system of different images

0.1.0 (2018-02-25)
++++++++++++++++++

* Major design update

0.1.1 (2018-03-05)
++++++++++++++++++

* minor update to facilitate options without lensing

0.2.0 (2018-03-10)
++++++++++++++++++

* ellipticity parameter handling changed
* time-delay distance sampling included
* parameter handling for sampling more flexible
* removed redundancies in the light and mass profiles

0.2.1 (2018-03-19)
++++++++++++++++++

* updated documentation
* improved sub-sampling of the PSF

0.2.2 (2018-03-25)
++++++++++++++++++

* improved parameter handling
* minor bugs with parameter handling fixed

0.2.8 (2018-05-31)
++++++++++++++++++

* improved GalKin module
* minor improvements in PSF reconstruction
* mass-to-light ratio parameterization

0.3.1 (2018-07-21)
++++++++++++++++++

* subgrid psf sampling for inner parts of psf exclusively
* minor stability improvements
* cleaner likelihood definition
* additional Chameleon lens and light profiles

0.3.3 (2018-08-21)
++++++++++++++++++
* minor updates, better documentation and handling of parameters

0.4.1-3 (2018-11-27)
++++++++++++++++++++
* various multi-band modelling frameworks added
* lens models added
* Improved fitting sequence, solver and psf iteration

0.5.0 (2019-1-30)
+++++++++++++++++
* Workflow module redesign
* improved parameter handling
* improved PSF subsampling module
* relative astrometric precision of point sources implemented

0.6.0 (2019-2-26)
+++++++++++++++++
* Simulation API module for mock generations
* Multi-source plane modelling

0.7.0 (2019-4-13)
+++++++++++++++++
* New design of Likelihood module
* Updated design of FittingSequence
* Exponential Shapelets implemented

0.8.0 (2019-5-23)
+++++++++++++++++
* New design of Numerics module
* New design of PSF and Data module
* New design of multi-band fitting module

0.8.1 (2019-5-23)
+++++++++++++++++
* PSF numerics improved and redundancies removed.

0.8.2 (2019-5-27)
+++++++++++++++++
* psf_construction simplified
* parameter handling for catalogue modelling improved

0.9.0 (2019-7-06)
+++++++++++++++++
* faster fft convolutions
* re-design of multi-plane lensing module
* re-design of plotting module
* nested samplers implemented
* Workflow module with added features

0.9.1 (2019-7-21)
+++++++++++++++++
* non-linear solver for 4 point sources updated
* new lens models added
* updated Workflow module
* implemented differential extinction

0.9.2 (2019-8-29)
+++++++++++++++++
* non-linear solver for 4 point sources updated
* Moffat PSF for GalKin in place
* Likelihood module for point sources and catalogue data improved
* Design improvements in the LensModel module
* minor stability updates

0.9.3 (2019-9-25)
+++++++++++++++++
* improvements in SimulationAPI design
* improvements in astrometric uncertainty handling of parameters
* local arc lens model description and differentials


1.0.0 (2019-9-25)
+++++++++++++++++
* marking version as 5 - Stable/production mode

1.0.1 (2019-10-01)
++++++++++++++++++
* compatible with emcee 3.0.0
* removed CosmoHammer MCMC sampling

1.1.0 (2019-11-5)
+++++++++++++++++
* plotting routines split in different files
* curved arc parameterization and eigenvector differentials
* numerical differentials as part of the LensModel core class


1.2.0 (2019-11-17)
++++++++++++++++++
* Analysis module re-designed
* GalKin module partially re-designed
* Added cosmography module
* parameterization of cartesian shear coefficients changed


1.2.4 (2020-01-02)
++++++++++++++++++
* First implementation of a LightCone module for numerical ray-tracing
* Improved cosmology sampling from time-delay cosmography measurements
* TNFW profile lensing potential implemented


1.3.0 (2020-01-10)
++++++++++++++++++
* image position likelihood description improved


1.4.0 (2020-03-26)
++++++++++++++++++
* Major re-design of GalKin module, added new anisotropy modeling and IFU aperture type
* Updated design of the Analysis.kinematicsAPI sub-module
* Convention and redundancy in the Cosmo module changed
* NIE, SIE and SPEMD model consistent with their ellipticity and Einstein radius definition
* added cored-Sersic profile
* dependency for PSO to CosmoHammer removed
* MPI and multi-threading for PSO and MCMC improved and compatible with python3


1.5.0 (2020-04-05)
++++++++++++++++++
* Re-naming SPEMD to PEMD, SPEMD_SMOOTH to SPEMD
* adaptive numerics improvement
* multi-processing improvements


1.5.1 (2020-06-20)
++++++++++++++++++
* bug fix in Hession of POINT_SOURCE model
* EPL model from Tessore et al. 2015 implemented
* multi-observation mode for kinematics calculation


1.6.0 (2020-09-07)
++++++++++++++++++
* SLITronomy integration
* observation configuration templates and examples
* lens equation solver arguments in single sub-kwargs
* adapted imports to latest scipy release