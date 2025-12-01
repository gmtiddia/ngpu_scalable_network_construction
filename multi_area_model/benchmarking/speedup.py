import numpy as np
import pandas as pd

data = pd.read_csv("processed_times_for_comparison_plot.csv")

data = data[data["state"]=="metastable"]

timers = ['time_prepare', 'time_create_nodes', 'time_connect_local', 'time_connect_global', 'time_calibrate', 'total_constr', 'time_simulate']

onboard = data[data["simulator"]=="onboard"]
offboard = data[data["simulator"]=="offboard"]

onboard = onboard[timers]
offboard = offboard[timers]

print("Speedup factors")
print(offboard.mean()/onboard.mean())

print("\nNEST GPU offboard mean and std")
print(offboard.mean())
print(offboard.std())

print("\nNEST GPU onboard mean and std")
print(onboard.mean())
print(onboard.std())

print("\nNEST GPU offboard median")
print(offboard.median())

print("\nNEST GPU onboard median")
print(onboard.median())



