# Description of the update of the lenstronomy

This repository contains modifications of lenstronomy based on the late August 2021 version of the original lenstronomy https://github.com/sibirrer/lenstronomy .

The purpose of this modification is to analyze the noise of the ALMA data by studying the non-diagonal noise pixel-pixel covariance matrix
and thus help to fit the lens and source model of a gravitational lens picture.

Here is the list of the updates based on the original lenstronomy: 
- 1. Add 'primary_beam' to the data_kwargs. Allow adding a primary beam after an image is generated but not convolved by PSF in both ImageModel and fitting.
- 2. Add 'use_linear_solver' to the data_kwargs, which (must be together with kwargs_constraints = {'linear_solver':False/True}) controls whether we perform a linear solver to find the optimal 'amp's in fitting.
- 3. Add two more methods to calculate the log_likelihood for interferometry data using the natwt PSF and the corresponding covariance matrix. Use kwargs_data['likelihood_method'] to tell the fitting program which method to use. For details reading the comments in lenstronomy/Data/imaging_data.py  .

Some example jupyter notebooks:

 - [Noise covariance matrix, eigen values and eigen modes -- a square sampling region](https://github.com/nanz6/lenstronomy_learning_notebook/blob/main/Noise%20covariance%20matrix%2C%20eigen%20values%20and%20eigen%20modes%20--%20a%20square%20sampling%20region.ipynb)
