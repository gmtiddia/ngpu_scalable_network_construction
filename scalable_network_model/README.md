# Scalable network model
The following directory contains the scripts to assess NEST GPU weak scaling performance using our scalable network model.

```av3_run_plot.py``` is the Python script that reproduces Figures 2, 3 and 4 of the manuscript.
The flag ```scale``` can be passed to obtain the plots of the same experiment run at scales 10, 20 and 30.
Currently, only these values can be used. 

The figures produced by the script are saved as PDF files in the directories ```results_scale_[scale]```,
in which the compressed data of the simulations performed and their collected timers
(i.e., ```aggregated_data_scale[scale].json``` and ```barplot_data_scale[scale].csv```) are also stored.

```indegree_scaling_plot.py``` is the Python script that reproduces Figure 12. Data needed for running the script is 
contained in the ```change_indegree``` directory.
