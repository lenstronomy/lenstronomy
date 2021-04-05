---
title: 'lenstronomy II: A gravitational lensing software ecosystem'
tags:
  - Python
  - astronomy
  - gravitational lensing
  - image simulations
  - dynamics

authors:
  - name: Simon Birrer^[sibirrer@stanford.edu]
    orcid: 0000-0003-3195-5507
    affiliation: "1, 2"
  - name: Author Without ORCID
    affiliation: 2
  - name: Author with no affiliation
    affiliation: 3
affiliations:
 - name: Kavli Institute for Particle Astrophysics and Cosmology and Department of Physics, Stanford University, Stanford, CA 94305, USA
   index: 1
 - name: SLAC National Accelerator Laboratory, Menlo Park, CA, 94025, USA
   index: 2
 - name: Independent Researcher
   index: 3
date: 2 April 2021
bibliography: paper.bib

<!--- 
# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
-->

---

# Summary

`lenstronomy` is an Astropy-affiliated Python package for gravitational lensing simulations and analyses.
Originally introduced by `@lenstronomyPDU` and based on the linear basis set approach by `@Birrer:2015`,
the user and developer base of `lenstronomy` has substantially grown 
and the software has become an integral part for a wide range of recent analyses, such as
measuring the Hubble constant with time-delay strong lensing or constraining the nature of Dark Matter from resolved and unresolved
small scale lensing distortion statistics. The modular design allowed the community to incorporate innovative
new methods, as well as developing enhanced software and wrappers with more specific aims on top of the `lenstronomy` API.
Through the community engagement and involvement, `lenstronomy` has become a foundation of an ecosystem of affiliated packages
extending the original scope of the software and proving its robustness and applicability 
at the forefront of the strong gravitational lensing community, open source and reproducible.


<!--- 
perhaps thsi figure needs to be changed as it appeared already in the previous publication.
-->

# Background
Gravitational lensing displaces the observed position and distorts the shape of apparent objects on the sky due to intervening inhomogeneous matter along the line of sight.
Strong gravitational lensing describes the regime where the background source, such as a galaxy or quasar, is lensed by a massive foreground object, such as another galaxy or cluster of galaxies,
to produce multiple images of itself in a highly distorted manner.
Figure \autoref{fig:example} illustrates such a process from the intrinsic galaxy to the data product at hand, including, besides the lensing distortions,
effects of the instrument, observational conditions and noise.
Analyses of strong gravitational lensing have provided a wealth of key insights into cosmology and astrophysics. 
For example, relative time delays of multiply imaged variable sources provided precision measurements on 
the expansion rate of the Universe `[@Wong:2020, @Shajib:strides, @Birrer:2020tdcosmoiv]`. 
Small scale distortions in the lensing signal of resolved sources `[@Vegetti:2012, @Hezaveh:2016, @Birrer:2017]`
and unresolved flux ratios `[@Gilman:2020, @Hsueh:2020]` constrain the nature of Dark Matter.
Combined strong lensing and kinematic observables constrain the formation and evolution of galaxies `[@Sonnenfeld:2015, @Shajib:slacs]`, 
and the magnification effect provides an otherwise inaccessible angle on the early universe (citatons?).





![Illustration of the strong gravitational lensing phenomenology. From left to right:
A galaxy is lensed around o foreground massive object, becomes highly distorted having components appearing multiple times. Observations of this phenomena are limited in resolution (convolution), depending on the detector (pixelisation) and are subject to noise.\label{fig:example}](paper_fig.png)

# Statement of need

Studies utilizing strong gravitational lensing have significantly enhance, and sometimes challenge and put in scrutiny, 
our current fundamental understanding of the Universe.
In the near future, with the onset of the next generation ground and 
space-based wide and deep astronomical imaging (Rubin, Roman, Euclid observatories) 
and interferrometric (SKA) surveys, the number of lenses
It is key that these demanding studies, their measurements and conclusions, 
are supported by reproducible and open-source available analysis products and software 
to provide the most compelling and transparent evidence required to transform our physical understanding.



- reliable specific functionality, adaptive and user-guided analysis settings satisfying specific needs in the analysis.
- evolving requirements on robustness of analyses

`lenstronomy` has demonstrated
- used for the analysis of more than 50+lenses `[@Shajib:2018, @Shajib:2020slacs]`, 30+peer reviewed publications

- tested on TDLMC (three participating teams)

- even more so in the future with automatization 

- affiliated packages




# Examples, prior usage, and affiliated packages
- innovations `[@Birrer:2015; @Tessore:2015; @Shajib:2019; @Joseph:2019; @Galan:2020]`
- end-to-end analyses

- due to high design requirements applications expand to other fields




# Acknowledgements

X is supported by Z

# References