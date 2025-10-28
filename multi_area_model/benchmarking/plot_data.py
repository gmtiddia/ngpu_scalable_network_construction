import numpy as np
import pandas as pd
import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tol_colors
from matplotlib.ticker import ScalarFormatter, FuncFormatter, NullFormatter, FixedLocator

try:
    import bennchplot as bp
    from matplotlib.patches import Rectangle, ConnectionPatch
    BENNCHPLOT_AVAILABLE = True
except ImportError:
    print("Warning: 'bennchplot' library not found. The 'bennchplot' and 'merged' plot options will not be available.")
    BENNCHPLOT_AVAILABLE = False

# plot style configuration
size_factor = 1.3
matplotlib_params = {
    'image.origin': 'lower', 'image.interpolation': 'nearest', 'axes.grid': False,
    'axes.labelsize': 15 * size_factor, 'axes.titlesize': 19 * size_factor,
    'font.size': 16 * size_factor, 'legend.fontsize': 13 * size_factor,
    'xtick.labelsize': 13 * size_factor, 'ytick.labelsize': 13 * size_factor,
    'text.usetex': False,
}
plt.rcParams.update(matplotlib_params)

bright, vibrant, light = tol_colors.colorsets['bright'], tol_colors.colorsets['vibrant'], tol_colors.colorsets['light']

def merge_logfile_data(path: str, nareas: int = 32):
    """
    Merges individual JSON log files for each simulation area into a single CSV file.
    This is a preprocessing step for `process_sim_data`.

    Args:
        path (str): The directory containing a single simulation's results.
        nareas (int): The number of simulation areas (and thus log files) to merge.
    """
    if not os.path.isdir(path): return
    sim_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d, "recordings"))]
    for dir in sim_dirs:
        merged_file = os.path.join(path, dir, "recordings", f"{dir}_merged.csv")
        if os.path.isfile(merged_file): continue
        print(f"Merging data for simulation {dir}")
        sim_res = pd.DataFrame({})
        for area in range(nareas):
            fn = os.path.join(path, dir, "recordings", f"{dir}_logfile_{area}")
            try:
                with open(fn) as f:
                    area_logfile = pd.DataFrame(json.loads(f.read()), index=[area])
                    area_logfile['total_constr'] = sum(area_logfile[c] for c in ['time_prepare', 'time_network_local_tot', 'time_connect_global', 'time_calibrate'])
                    sim_res = pd.concat([sim_res, area_logfile])
            except FileNotFoundError: pass
        sim_res.to_csv(merged_file)

def process_sim_data(path: str, nsim: int, method: str, nareas: int):
    """
    Processes the merged CSVs for multiple simulation runs in a directory.
    It averages the timing data across MPI processes for each run.

    Args:
        path (str): The top-level directory containing multiple simulation runs.
        nsim (int): The number of simulation runs to process.
        method (str): The averaging method (currently only 'mean' is supported).
        nareas (int): The number of areas, passed down to `merge_logfile_data`.

    Returns:
        pd.DataFrame: A DataFrame containing the averaged results for each simulation run.
    """
    merge_logfile_data(path=path, nareas=nareas)
    sim_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d, "recordings"))]
    if not sim_dirs: return pd.DataFrame()
    nsim = min(nsim, len(sim_dirs))
    all_sim_data = pd.DataFrame({})
    for dir in sim_dirs[:nsim]:
        sim_data_path = os.path.join(path, dir, "recordings", f"{dir}_merged.csv")
        sim_data = pd.read_csv(sim_data_path, index_col=0)
        processed_data = sim_data.mean(axis=0).to_dict()
        processed_data["sim_label"] = dir
        all_sim_data = pd.concat([all_sim_data, pd.DataFrame(processed_data, index=[len(all_sim_data)])])
    return all_sim_data

def prepare_data_for_original_plot(paths: list, labels: list, sim_versions: list, nareas: list, nsim: int):
    """
    Main data preparation pipeline for the "original" two-panel plot.
    It orchestrates the processing of raw log files from multiple directories,
    calculates derived metrics, and saves the final data to a CSV file.

    Args:
        paths (list): A list of directories to process.
        labels (list): A list of state labels ('ground', 'metastable') corresponding to the paths.
        sim_versions (list): A list of simulator versions ('onboard', 'offboard') corresponding to the paths.
        nareas (list): A list of area counts corresponding to the paths.
        nsim (int): The number of simulations to process from each directory.

    Returns:
        pd.DataFrame: The fully processed DataFrame ready for plotting.
    """
    print("Processing log files for the 'comparison' plot...")
    data = pd.DataFrame({})
    for i, p in enumerate(paths):
        dum = process_sim_data(path=p, nsim=nsim, method='mean', nareas=nareas[i])
        if dum.empty: continue
        # Convert times from nanoseconds to seconds
        dum = dum.drop(["sim_label"], axis=1, errors='ignore') / 1e9
        # Add metadata and calculate derived metrics
        dum["state"], dum["simulator"], dum["model_time_sim"] = labels[i], sim_versions[i], 10.0
        dum['time_create_nodes'] = dum['time_create_neurons'] + dum['time_create_devices']
        dum['time_connect_local'] = dum['time_connect_local'] + dum['time_connect_devices']
        dum['sim_factor'] = dum['time_simulate'] / dum['model_time_sim']
        data = pd.concat([data, dum])
    data = data.reset_index(drop=True)
    
    output_filename = "processed_times_for_comparison_plot.csv"
    data.to_csv(output_filename, index=False)
    print(f"Data for original plot saved to '{output_filename}'")
    return data

# plotting 
def plot_comparison(data: pd.DataFrame, state: str = "metastable"):
    """
    Generates the standalone two-panel "version comparison" plot.

    Args:
        data (pd.DataFrame): The DataFrame from `prepare_data_for_original_plot`.
        state (str): The simulation state to plot ('ground' or 'metastable').
    """
    print(f"Generating original style plot for '{state}' state...")
    data_subset = data[data["state"] == state].copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), constrained_layout=True)
    plot_comparison_panels(ax1, ax2, data_subset)
    plt.suptitle(f"Original Style Plot ({state} state)", fontsize=24 * size_factor)
    plt.savefig("version_comparison_plot.png", dpi=300); plt.show()

def plot_scaling(bennchplot_filename: str):
    """
    Generates the standalone three-panel "scaling" plot using the bennchplot library.

    Args:
        bennchplot_filename (str): Path to the pre-processed CSV file for scaling analysis.
    """
    if not BENNCHPLOT_AVAILABLE: print("Cannot generate bennchplot plot."); return
    print(f"Generating bennchplot style plot from '{bennchplot_filename}'...")
    
    args = {'data_file': bennchplot_filename, 'x_axis': ['nodes'], 'time_scaling': 1.0}
    B = bp.Plot(**args)

    fig = plt.figure(figsize=(13.5, 7), constrained_layout=True)
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, width_ratios=[1, 1], height_ratios=[3, 1])
    ax1, ax2, ax3 = fig.add_subplot(spec[:, 0]), fig.add_subplot(spec[0, 1]), fig.add_subplot(spec[1, 1])

    for ax, label in zip([ax1, ax2, ax3], ["a", "b", "c"]):
        ax.text(-0.15, 1.05 if label != "C" else 1.4, label, fontsize=25, weight="bold", transform=ax.transAxes)
    
    B.plot_fractions(axis=ax1, fill_variables=['total_constr', 'time_simulate'], interpolate=True, step=None, error=True)
    construction_vars = ['time_configure', 'time_area_packing', 'time_create_nodes', 'time_connect_local', 'time_connect_global', 'time_calibrate']
    B.plot_fractions(axis=ax2, fill_variables=construction_vars, interpolate=True, step=None, error=True)
    B.plot_main(quantities=['sim_factor'], axis=ax3, error=True, fmt='-', ecolor='black')

    for ax in [ax1, ax2, ax3]: B.simple_axis(ax)
    ax1.set_xlabel('Number of nodes'); ax1.set_ylabel(r'$T_{\mathrm{wall}}$ [s]'); ax1.set_yscale('log')
    ax2.set_xlabel('Number of nodes'); ax2.set_ylabel('Network construction [s]'); ax2.set_ylim(0, 150)
    ax3.set_xlabel('Number of nodes'); ax3.set_ylabel(r'$T_{\mathrm{wall}} / T_{\mathrm{model}}$')

    h, l = ax1.get_legend_handles_labels(); ax1.legend(h[::-1], l[::-1])
    h, l = ax2.get_legend_handles_labels(); ax2.legend(h[::-1], l[::-1], loc='upper right', bbox_to_anchor=[1.55, 0.8], ncols=1)
    
    plt.suptitle("Bennchplot Style Plot", fontsize=24 * size_factor)
    plt.savefig("scaling_plot.png", dpi=300); plt.show()

def plot_merged(original_plot_df: pd.DataFrame, bennchplot_filename: str, state: str = "metastable"):
    """
    Generates the final 5-panel manuscript figure, merging the two analyses.

    Args:
        original_plot_df (pd.DataFrame): Data for the top two panels.
        bennchplot_filename (str): Path to the data file for the bottom three panels.
        state (str): The simulation state to display in the top panels.
    """
    if not BENNCHPLOT_AVAILABLE: print("Cannot generate merged plot."); return
    print(f"Generating merged 5-panel plot for '{state}' state...")

    # adjust figure size here
    fig = plt.figure(figsize=(18, 14))

    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    axA = fig.add_subplot(spec[0, 0])
    axB = fig.add_subplot(spec[0, 1])
    axC = fig.add_subplot(spec[1, 0])

    subspec = spec[1, 1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.4) # KNOB: Inner space D-E
    axD = fig.add_subplot(subspec[0])
    axE = fig.add_subplot(subspec[1])

    # plotting panels
    # Panels A & B (NEST GPU version comparison)
    data_subset = original_plot_df[original_plot_df["state"] == state].copy()
    plot_comparison_panels(axA, axB, data_subset)
    # Panels C, D, E (scaling)
    B = bp.Plot(data_file=bennchplot_filename, x_axis=['nodes'], time_scaling=1.0)
    construction_vars = ['time_configure', 'time_area_packing', 'time_create_nodes', 'time_connect_local', 'time_connect_global', 'time_calibrate']
    B.plot_fractions(axis=axC, fill_variables=construction_vars, interpolate=True, step=None, error=True)
    B.plot_fractions(axis=axD, fill_variables=['total_constr', 'time_simulate'], interpolate=True, step=None, error=True)
    B.plot_main(axis=axE, quantities=['sim_factor'], error=True, fmt='-', ecolor='black')

    # Panel C inset
    axins = axC.inset_axes([0.5, 0.5, 0.45, 0.4]) # KNOB: Inset size/position
    B.plot_fractions(axis=axins, fill_variables=construction_vars, interpolate=True, step=None, error=True)
    if axins.get_legend() is not None: axins.get_legend().remove()
    x1_zoom, x2_zoom = 1, 9
    axins.set_xlim(x1_zoom, x2_zoom); axins.set_yscale('log')
    axins.set_xlabel(''); axins.set_ylabel('')
    axins.tick_params(axis='both', which='major', labelsize=10*size_factor)

    y1_box, y2_box = 0, 105
    rect = Rectangle((x1_zoom, y1_box), x2_zoom - x1_zoom, y2_box - y1_box, edgecolor="gray", linestyle='-', facecolor="none", zorder=2)
    axC.add_patch(rect)
    con1 = ConnectionPatch(xyA=(0, 1), xyB=(x1_zoom, y2_box), coordsA="axes fraction", coordsB="data", axesA=axins, axesB=axC, color="silver", linestyle='-')
    con2 = ConnectionPatch(xyA=(0, 0), xyB=(x1_zoom, y1_box), coordsA="axes fraction", coordsB="data", axesA=axins, axesB=axC, color="silver", linestyle='-')
    fig.add_artist(con1); fig.add_artist(con2)

    # legend
    handles_C, labels_C = axC.get_legend_handles_labels()
    if axC.get_legend() is not None: axC.get_legend().remove()
    if axA.get_legend() is not None: axA.get_legend().remove()
    axA.legend(handles_C[::-1], labels_C[::-1], fontsize=13*size_factor, loc='best')

    # panel labels
    all_axes = [axA, axB, axC, axD, axE]
    for ax, label in zip(all_axes, ["a", "b", "c", "d", "e"]):
        ax.text(-0.135, 1.05, label, fontsize=30, weight="bold", transform=ax.transAxes)
        ax.grid(True)
        B.simple_axis(ax)
    
    # subtitles
    fig.text(0.5, 0.975, "Version comparison", ha='center', va='bottom', fontsize=19*size_factor)
    fig.text(0.5, 0.46, "NEST GPU onboard scaling experiment", ha='center', va='bottom', fontsize=19*size_factor)

    axC.set_xlabel('Number of nodes'); axC.set_ylabel('Network construction [s]')
    axC.set_ylim(0, 150)
    axD.set_ylabel('Time [s]')

    axD.tick_params(axis='x', labelbottom=False)
    
    axD.set_yscale('log')

    axD.set_yticks([60, 100, 200, 300, 400])

    axD.yaxis.set_major_locator(FixedLocator([60, 100, 200, 300, 400]))
    axD.yaxis.minorticks_off()

    current_top_limit = axD.get_ylim()[1]
    axD.set_ylim(bottom=50, top=current_top_limit)
    
    axD.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
    axD.yaxis.set_minor_formatter(NullFormatter())
    
    h, l = axD.get_legend_handles_labels(); axD.legend(h[::-1], l[::-1], fontsize=13*size_factor)
    
    axE.set_xlabel('Number of nodes'); axE.set_ylabel(r'$T_{\mathrm{wall}} / T_{\mathrm{model}}$')
    
    if axE.get_legend() is not None: axE.get_legend().remove()

    # adjust spacing
    plt.subplots_adjust(hspace=0.4, left=0.07, right=0.97, top=0.93, bottom=0.05)
    plt.savefig("merged_plot.png", dpi=300); plt.show()

def plot_comparison_panels(ax1, ax2, data_subset):
    """
    Helper function to draw the two comparison panels on a given pair of axes.
    This is used by both `plot_comparison_panels` and `plot_merged`.
    
    Args:
        ax1 (matplotlib.axes.Axes): The axis for the stacked bar plot.
        ax2 (matplotlib.axes.Axes): The axis for the box plot.
        data_subset (pd.DataFrame): The data filtered for the correct state.
    """
    # stacked bar plot for construction time
    stacked_constr = pd.DataFrame({
        "simulator": data_subset["simulator"], "time_initialize": data_subset["time_prepare"], "time_create_nodes": data_subset["time_create_nodes"],
        "time_connect_local": data_subset["time_connect_local"], "time_connect_remote": data_subset["time_connect_global"], "time_calibrate": data_subset["time_calibrate"]
    })
    
    stacked_constr["time_create_nodes_cum"] = stacked_constr["time_create_nodes"] + stacked_constr["time_initialize"]
    stacked_constr["time_connect_local_cum"] = stacked_constr["time_connect_local"] + stacked_constr["time_create_nodes_cum"]
    stacked_constr["time_connect_remote_cum"] = stacked_constr["time_connect_remote"] + stacked_constr["time_connect_local_cum"]
    stacked_constr["time_calibrate_cum"] = stacked_constr["time_calibrate"] + stacked_constr["time_connect_remote_cum"]

    error_bar_properties = {'color': 'black', 'linewidth': 1.}
    
    sns.barplot(ax=ax1, data=stacked_constr, x="simulator", y="time_calibrate_cum", label="Calibration", color=light.light_yellow, errorbar='sd', capsize=0.015, err_kws=error_bar_properties)
    sns.barplot(ax=ax1, data=stacked_constr, x="simulator", y="time_connect_remote_cum", label="Remote connection", color=light.mint, errorbar='sd', capsize=0.015, err_kws=error_bar_properties)
    sns.barplot(ax=ax1, data=stacked_constr, x="simulator", y="time_connect_local_cum", label="Local connection", color=bright.green, errorbar='sd', capsize=0.015, err_kws=error_bar_properties)
    sns.barplot(ax=ax1, data=stacked_constr, x="simulator", y="time_create_nodes_cum", label="Node creation", color=light.light_blue, errorbar='sd', capsize=0.015, err_kws=error_bar_properties)
    sns.barplot(ax=ax1, data=stacked_constr, x="simulator", y="time_initialize", label="Initialization", color=vibrant.orange, errorbar='sd', capsize=0.015, err_kws=error_bar_properties)
    ax1.set_xlabel('NEST GPU version'); ax1.set_ylabel('Network construction [s]')
    ax1.legend()

    # box plot for rtf
    sns.boxplot(ax=ax2, data=data_subset, x="simulator", y="sim_factor", linewidth=2.5, fill=False, color=light.pink,
                flierprops=dict(marker='o', markerfacecolor=light.pink, markersize=7, markeredgecolor=light.pink))
    ax2.set_xlabel('NEST GPU version'); ax2.set_ylabel(r'$T_{\mathrm{wall}} / T_{\mathrm{model}}$')

# main execution block
if __name__ == "__main__":
    # name of the csv file collecting data
    BENNCHPLOT_DATA_FILE = "data_strong_scaling/processed_times_mean.csv"
    
    # set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Generate benchmark plots from simulation data.")
    parser.add_argument('--plot-type', type=str, choices=['comparison', 'scaling', 'merged'], default='merged', help='The type of plot to generate.')
    parser.add_argument('--state', type=str, choices=['ground', 'metastable'], default='metastable', help="Simulation state for categorical plots.")
    parser.add_argument('--nsim', type=int, default=10, help="Number of simulations to process for the 'comparison' plot.")
    args = parser.parse_args()

    # logic to call the correct plotting function
    if args.plot_type == 'scaling':
        if not os.path.exists(BENNCHPLOT_DATA_FILE): print(f"Error: Bennchplot data file '{BENNCHPLOT_DATA_FILE}' not found.")
        else: plot_scaling(BENNCHPLOT_DATA_FILE)

    elif args.plot_type in ['comparison', 'merged']:
        # define paths and metadata for data processing
        base_path = "./data_comparison/"
        paths = [os.path.join(base_path, p) for p in ["main/simulations_gs/", "main/simulations_ms/", "mpi_comm/simulations_gs/", "mpi_comm/simulations_ms/"]]
        labels, sim_versions, nareas = ['ground', 'metastable', 'ground', 'metastable'], ['offboard', 'offboard', 'onboard', 'onboard'], [32, 32, 32, 32]
        
        # step required for 'comparison' and 'merged' plots
        #original_plot_df = prepare_data_for_original_plot(paths=paths, labels=labels, sim_versions=sim_versions, nareas=nareas, nsim=args.nsim)
        original_plot_df = pd.read_csv('processed_times_for_comparison_plot.csv')

        if args.plot_type == 'comparison':
            plot_comparison(original_plot_df, state=args.state)
        elif args.plot_type == 'merged':
            if not os.path.exists(BENNCHPLOT_DATA_FILE): print(f"Error: Bennchplot data file '{BENNCHPLOT_DATA_FILE}' not found.")
            else: plot_merged(original_plot_df, BENNCHPLOT_DATA_FILE, state=args.state)


