import numpy as np
import pandas as pd

data = pd.read_csv("processed_times_for_comparison_plot.csv")

data = data[data["state"]=="metastable"]

timers = ['time_prepare', 'time_create_nodes', 'time_connect_local', 'time_connect_global', 'time_calibrate', 'total_constr']

onboard = data[data["simulator"]=="onboard"]
offboard = data[data["simulator"]=="offboard"]

onboard = onboard[timers]
offboard = offboard[timers]

print(offboard.mean()/onboard.mean())



