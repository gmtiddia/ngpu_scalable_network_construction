# Scalable network model

The following directory contains the scripts for assessing NEST GPU weak scaling performance in the simulation of the scalable network model. The following provide a more detailed description of the content.

```run_plot.py``` is the Python script that reproduces Figures X, Y and Z of the manuscript. The flag ```scale``` can be passed to obtain the plots of the same experiment run at scales 10, 20 (the one of the main body of the manuscript) and 30. Only these tree values can be used. 

The figures produced by the script are saved into PDF files in the directories 
```results_scale_[scale]```, in which the compressed data of the simulations performed and their collected timers (i.e., ```aggregated_data_scale[scale].json``` and ```barplot_data_scale[scale].csv```) are also stored.







