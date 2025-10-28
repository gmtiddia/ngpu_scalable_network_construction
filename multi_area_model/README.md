# Multi-Area Model

The following directory contains the scripts for validating the NEST GPU simulator and to assess its performance in the simulation of the Multi-Area Model. The following provide a more detailed description of the content of the directories.


## Benchmarking

The [benchmarking](benchmarking/) directory contains the simulation data and the plotting scripts of the comparison of the performance of the two NEST GPU versions and of the assessment of the strong scaling performance of the onboard version.

### Contents
The [data_comparison](benchmarking/data_comparison/) directory contains two compressed files of the sets of simulations used to compare the two versions of NEST GPU.

The [data_strong_scaling](benchmarking/data_strong_scaling/) directory contains a compressed file of the simulations performed to assess the strong scaling performance. 
```get_data.py``` is the Python script that, given the timers obtained in each simulation of the MAM, collects the results into two csv files, namely ```processed_times_mean.csv``` and ```averaged_data.csv```, which contain the timers for each simulation and the ones averaged over thge simulations, respectively. The two csv files are provided in the directory for convenience.

```plot_data.py``` is the Python script that reproduces Figure X of the manuscript. Moreover, it produces ```processed_times_for_comparison_plot.csv```, which results from the analysis of the simulations performed for the version comparison.


## Validation

The scripts needed to perform the validation protocol are contained into the [MAM validation](https://github.com/gmtiddia/ngpu_mam_validation.git) reopository.

In the [validation](validation/) directory, we provide the plots that results from the validation performed in the context of this work. The directory sample_plots returns the Figures X and Y of the Appendix Z of the manuscript, whereas 
in emd_all_areas are reported the values of the Earth Mover Distance collected for each area of the MAM for the three distributions of the spike statistics.


