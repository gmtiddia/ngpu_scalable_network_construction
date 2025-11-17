# NEST GPU Scalable Network Construction

This repository contains the data and scripts used for the preprint: *preprint name*.
<br>
If you want to cite this, use:
<br>
<br>
Tiddia, Villamar, Golosio, et al. 2025, [10.5281/zenodo.17627865](https://zenodo.org/records/17627865)
<br>

## Requirements
To run the simulations, NEST GPU version [offboard]() and version [onboard]() are required.

For installation instructions on the simulators, see [NEST GPU Documentation](https://nest-gpu.readthedocs.io/en/latest/installation/index.html).

Additionally to run the scripts to post process the data and generate plots, Python and additional packages are required.
To run the data post processing scripts and plotting scripts the following software was used:
 * Python 3.8.6
 * Pandas 1.3.3
 * Numpy 1.22
 * Matplotlib 3.5
 * Tol Colors 1.2.1 (https://pypi.org/project/tol-colors/)
 * [beNNch-plot](https://github.com/gmtiddia/beNNch-plot) forked to adapt the beNNch plotting style to NEST GPU benchmarking data.


## Contents
The [multi_area_model](multi_area_model/) directory contains benchmarking data and the scripts needed to produce the plots for the simulations of the Multi-Area Model. Additionally, it contains material and instructions to perform the validation.

The [scalable_network_model](scalable_network_model/) directory contains the scripts to generate the plots of the results of the simulations of the scalable spiking network model employed to assess the weak scaling performance of NEST GPU.



## Contact
Gianmarco Tiddia, Department of Physics, University of Cagliary, Italy, Istituto Nazionale di Fisica Nucleare, Sezione di Cagliari, Italy, gianmarco.tiddia@ca.infn.it
<br>
José Villamar, Institute for Advanced Simulation (IAS-6), Jülich Research Centre, Jülich, Germany, j.villamar@fz-juelich.de


## License
GPL 3.0 [license](LICENSE)
