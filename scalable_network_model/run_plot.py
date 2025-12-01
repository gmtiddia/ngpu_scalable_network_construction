import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import matplotlib.colors as mcolors
import tol_colors as tc
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch

# color palette
tol_vibrant = [tc.light.orange, tc.vibrant.cyan, tc.pale.pale_red, tc.light.mint, tc.pale.pale_green, tc.vibrant.red, tc.vibrant.teal, tc.light.pale_grey]
darker = ['#f06537', '#1395eb', '#faa2a2', tc.bright.green, tc.pale.pale_green, tc.vibrant.red, tc.vibrant.teal, tc.light.pale_grey]

tol_vibrant_rgb = [mcolors.to_rgb(c) for c in tol_vibrant]
color_map = dict(zip(tol_vibrant_rgb, darker))

def darken_color(rgb_color):
    """
    Finds the darker counterpart for a given RGB color tuple using a pre-built map.
    
    :param rgb_color: An RGB tuple (e.g., from matplotlib's color cycle).
    :return: The corresponding darker color's hex string.
    """
    
    key = tuple(val for val in rgb_color)
    
    return color_map.get(key, 'black')

sns.set_palette(palette=tol_vibrant)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# matplotlib settings
FONT_SCALE = 1.3
matplotlib_params = {
    'image.origin': 'lower', 'image.interpolation': 'nearest', 'axes.grid': False,
    'axes.labelsize': 15 * FONT_SCALE, 'axes.titlesize': 19 * FONT_SCALE,
    'font.size': 16 * FONT_SCALE, 'legend.fontsize': 11 * FONT_SCALE,
    'xtick.labelsize': 13 * FONT_SCALE, 'ytick.labelsize': 13 * FONT_SCALE,
    'text.usetex': False,
}
plt.rcParams.update(matplotlib_params)


parser = argparse.ArgumentParser(description="Generate benchmark plots from simulation data.")
parser.add_argument('--scale', type=int, default=20, help="Scale parameter (s) of the scalable network model. (default=20)")
args = parser.parse_args()



# lists needed to collect the data
# number of nodes
# scale parameter (s)
scale = args.scale

allowed_scales = [10, 20, 30]

if scale not in allowed_scales:
    raise ValueError(f"invalid scale parameter: {scale}, allowed values are {', '.join(map(str, allowed_scales))}")

real_nodes = [32, 64, 96, 128, 192, 256]
if scale != 30:
    fake_nodes = [32, 64, 96, 128, 192, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]
else:
    fake_nodes = [32, 64, 96, 128, 192, 256, 512, 1024]
# optimization levels
opt_levels_all = [-1, 0, 1, 2, 3]
opt_levels_partial = [-1, 0, 1, 2]
# seeds and ranks per node
run_seeds = list(range(5))
ranks_per_node = 4
# estimation of the number of total synapses
n_neurons_per_proc = 11250
indegree = 11250


synapses = [n_neurons_per_proc*indegree*ranks_per_node*scale*n for n in fake_nodes]
synapses_real = [n_neurons_per_proc*indegree*ranks_per_node*scale*n for n in real_nodes]


def format_synapses(total_conns):
        return fr"{total_conns / 1e12:.1f}"

def safe_mean(arr):
    return np.mean(arr) if arr else np.nan

def convert_keys_to_int(obj):
    "A function that converts a loaded json key to int."
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            try:
                new_key = int(k)
            except (ValueError, TypeError):
                new_key = k
            new_dict[new_key] = convert_keys_to_int(v)
        return new_dict
    elif isinstance(obj, list):
        return [convert_keys_to_int(i) for i in obj]
    else:
        return obj

# data loading
def load_and_aggregate_data(results_dir:str='./'):
    data = {}
    for o in opt_levels_all:
        data[o] = {
            "construct_avg": [], "construct_max": [], "construct_min": [], "construct_std": [],
            "calibrate_avg": [], "calibrate_max": [], "calibrate_min": [], "calibrate_std": [],
            "simulate_avg": [], "simulate_max": [], "simulate_min": [], "simulate_std": [],
            "gpu_mem_peak": [], "gpu_mem_peak_std": [], "gpu_mem_used": [], "gpu_mem_used_std": [],
            "fake_construct": [], "fake_calibrate": [], "fake_gpu_mem_peak": [], "fake_gpu_mem_used": [],
            "n_neurons": [], "n_connections": []
        }
        if o == 0:
            for N in fake_nodes:
                total_neurons = scale * ranks_per_node * N * n_neurons_per_proc
                total_conns = total_neurons * indegree
                data[o]["n_neurons"].append(total_neurons)
                data[o]["n_connections"].append(total_conns)
        for N in real_nodes:
            per_run_construct_avg, per_run_construct_max, per_run_construct_min = [], [], []
            per_run_calibrate_avg, per_run_calibrate_max, per_run_calibrate_min = [], [], []
            per_run_simulate_avg, per_run_simulate_max, per_run_simulate_min = [], [], []
            per_run_gpu_peak, per_run_gpu_used = [], []
            for r in run_seeds:
                construct_vals, calibrate_vals, simulate_vals = [], [], []
                gpu_peak_vals, gpu_used_vals = [], []
                folder = results_dir + f"nodes_{N}/opt_{o}_run_{r}"
                for rank in range(ranks_per_node * N):
                    file = os.path.join(folder, f"log_{rank}.json")
                    if os.path.exists(file):
                        with open(file, "r") as f: log = json.load(f)
                        t = log.get("timers", {}); g = log.get("gpu_mem", {})
                        construct_vals.append(t.get("time_construct", 0) / 1e9)
                        calibrate_vals.append(t.get("time_calibrate", 0) / 1e9)
                        simulate_vals.append(t.get("time_simulate", 0) / 1e9)
                        gpu_peak_vals.append(g.get("gpu_mem_peak", 0) / (2**30))
                        gpu_used_vals.append(g.get("gpu_mem_used", 0) / (2**30))
                if construct_vals:
                    per_run_construct_avg.append(np.mean(construct_vals)); per_run_construct_max.append(np.max(construct_vals)); per_run_construct_min.append(np.min(construct_vals))
                    per_run_calibrate_avg.append(np.mean(calibrate_vals)); per_run_calibrate_max.append(np.max(calibrate_vals)); per_run_calibrate_min.append(np.min(calibrate_vals))
                    per_run_simulate_avg.append(np.mean(simulate_vals)); per_run_simulate_max.append(np.max(simulate_vals)); per_run_simulate_min.append(np.min(simulate_vals))
                    per_run_gpu_peak.append(np.mean(gpu_peak_vals)); per_run_gpu_used.append(np.mean(gpu_used_vals))
            data[o]["construct_avg"].append(safe_mean(per_run_construct_avg)); data[o]["construct_max"].append(safe_mean(per_run_construct_max)); data[o]["construct_min"].append(safe_mean(per_run_construct_min))
            data[o]["calibrate_avg"].append(safe_mean(per_run_calibrate_avg)); data[o]["calibrate_max"].append(safe_mean(per_run_calibrate_max)); data[o]["calibrate_min"].append(safe_mean(per_run_calibrate_min))
            data[o]["simulate_avg"].append(safe_mean(per_run_simulate_avg)); data[o]["simulate_max"].append(safe_mean(per_run_simulate_max)); data[o]["simulate_min"].append(safe_mean(per_run_simulate_min))
            data[o]["gpu_mem_peak"].append(safe_mean(per_run_gpu_peak)); data[o]["gpu_mem_used"].append(safe_mean(per_run_gpu_used))
            data[o]["construct_std"].append(np.std(per_run_construct_avg) if per_run_construct_avg else np.nan); data[o]["calibrate_std"].append(np.std(per_run_calibrate_avg) if per_run_calibrate_avg else np.nan)
            data[o]["simulate_std"].append(np.std(per_run_simulate_avg) if per_run_simulate_avg else np.nan); data[o]["gpu_mem_peak_std"].append(np.std(per_run_gpu_peak) if per_run_gpu_peak else np.nan)
        for N in fake_nodes:
            folder = results_dir + f"nodes_{N}_fake/opt_{o}_run_0"
            construct_vals, calibrate_vals, gpu_peak_vals, gpu_used_vals = [], [], [], []
            for rank in range(4):
                file = os.path.join(folder, f"log_{rank}.json")
                if os.path.exists(file):
                    with open(file, "r") as f: log = json.load(f)
                    t = log.get("timers", {}); g = log.get("gpu_mem", {})
                    construct_vals.append(t.get("time_construct", 0) / 1e9); calibrate_vals.append(t.get("time_calibrate", 0) / 1e9)
                    gpu_peak_vals.append(g.get("gpu_mem_peak", 0) / (2**30)); gpu_used_vals.append(g.get("gpu_mem_used", 0) / (2**30))
            data[o]["fake_construct"].append(safe_mean(construct_vals)); data[o]["fake_calibrate"].append(safe_mean(calibrate_vals))
            data[o]["fake_gpu_mem_peak"].append(safe_mean(gpu_peak_vals)); data[o]["fake_gpu_mem_used"].append(safe_mean(gpu_used_vals))
        
        with open('results_scale_{}/aggregated_data_scale{}.json'.format(int(scale), int(scale)), 'w') as fp:
            json.dump(data, fp, indent=4)

    return data

# plotting function
def prepare_data_for_barplot(data_agg):
    """
    Creates a single long-form DataFrame containing both REAL and ESTIMATE data,
    ready for plotting with seaborn.
    
    Args:
        data_agg (dict): The aggregated data from load_and_aggregate_data().

    Returns:
        pd.DataFrame: A unified DataFrame for plotting.
    """

    plot_data = []
    quantities = ["Construct", "Calibrate"]

    data_agg = convert_keys_to_int(data_agg)

    for o in opt_levels_partial:
        for quantity_name in quantities:
            quantity_key = quantity_name.lower() # e.g., "construct"

            # add REAL data (from multiple runs)
            for i, N in enumerate(real_nodes):
                per_run_averages = []
                for r in run_seeds:
                    run_times = []
                    folder = f"results_scale_{scale}/nodes_{N}/opt_{o}_run_{r}"
                    for rank in range(ranks_per_node * N):
                        file = os.path.join(folder, f"log_{rank}.json")
                        if os.path.exists(file):
                            with open(file, "r") as f: log = json.load(f)
                            t = log.get("timers", {})
                            run_times.append(t.get(f"time_{quantity_key}", 0) / 1e9)
                    if run_times:
                        per_run_averages.append(np.mean(run_times))
                
                for avg_time in per_run_averages:
                    plot_data.append({
                        "Nodes": N,
                        "Time (s)": avg_time,
                        "Quantity": quantity_name,
                        "Type": "Real",
                        "Hue": f"Opt {o+1}"
                    })

            # add ESTIMATE data
            try:
                fake_avg_times = data_agg[o][f"fake_{quantity_key}"]
            except:
                o = str(o)
                fake_avg_times = data_agg[o][f"fake_{quantity_key}"]
            for i, N_fake in enumerate(fake_nodes):
                plot_data.append({
                    "Nodes": N_fake,
                    "Time (s)": fake_avg_times[i],
                    "Quantity": quantity_name,
                    "Type": "Estimate",
                    "Hue": f"Opt {o+1} Estimate"
                })

    plot_data = pd.DataFrame(plot_data)
    plot_data.to_csv(f"results_scale_{scale}/barplot_data_scale{scale}.csv")

    return plot_data


def plot_unified_time_as_barplot(df_unified, filename):
    """
    Creates a 1x2 plot from a unified DataFrame. "Estimate" data is shown
    as darker bars, and the mean of the "Real" data is shown as a single
    colored marker overlaid directly on top.
    
    Args:
        df_unified (pd.DataFrame): DataFrame from prepare_combined_data_for_barplot.
        filename (str): The output path for the saved figure.
    """

    fig, axs = plt.subplots(2, 1, figsize=(10, 13), sharey=False)
    quantities = ["Construct", "Calibrate"]
    quantities_labels = {"Construct": "Neuron and device\ncreation and connection", "Calibrate": "Preparation"}

    # palette for the "Estimate" bars
    bar_palette = {}
    for o in opt_levels_partial:
        bar_palette[f"Opt {o+1} Estimate"] = darken_color(colors[o + 1])

    # palette for the "Real" data mean markers
    marker_palette = {}
    for o in opt_levels_partial:
        marker_palette[f"Opt {o+1} Estimate"] = colors[o + 1]

    estimate_hues = [f"Opt {o+1} Estimate" for o in opt_levels_partial]
    
    panel_label = ["a", "b"]

    for col, quantity_name in enumerate(quantities):
        ax = axs[col]
        df_subset = df_unified[df_unified['Quantity'] == quantity_name]
        
        df_real = df_subset[df_subset['Type'] == 'Real'].copy()
        df_estimate = df_subset[df_subset['Type'] == 'Estimate']


        # plot estimate data
        sns.barplot(
            data=df_estimate, x="Nodes", y="Time (s)", hue="Hue", hue_order=estimate_hues,
            palette=bar_palette, ax=ax, edgecolor='black', linewidth=0.0, legend=False
        )

        df_real_mean = df_real.groupby(['Nodes', 'Hue'], as_index=False)['Time (s)'].mean()

        df_real_mean['Hue'] = df_real_mean['Hue'] + ' Estimate'
        
        sns.stripplot(
            data=df_real_mean,
            x="Nodes", 
            y="Time (s)", 
            hue="Hue", 
            hue_order=estimate_hues,
            #palette=marker_palette,
            color='k',
            ax=ax,
            dodge=True,
            jitter=False,
            marker='$-$',
            size=8,
            #edgecolor='black',
            #linewidth=1,
            legend=False
        )
        
        ax.text(-0.1, 1.0175, panel_label[col], fontsize=25, weight="bold", color='k', transform=ax.transAxes)
        ax.set_xlabel("Number of Nodes")# if col > 0 else "")
        ax.set_ylabel(f"{quantities_labels[quantity_name]} [s]")
        
        # legend
        if col == 0:
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            handles = []
            for o in opt_levels_partial:
                base_color = colors[o + 1]
                dark_color = darken_color(base_color)
                handles.append(Patch(facecolor=dark_color, edgecolor='black', linewidth=0., 
                                     label=f"Opt {o+1} Est."))

            handles.append(Line2D([0], [0], marker='$-$', color='k', linestyle='None',
                            markersize=8,
                            label=f"Sim. results"))
            ax.legend(handles=handles, ncol=1)

        
        tick_locations = ax.get_xticks()
        for tick in tick_locations[::2]:
            ax.axvspan(
                tick - 0.5, 
                tick + 0.5, 
                color='lightgray', 
                alpha=0.5,
                zorder=0
            )
        
        left_limit = tick_locations[0] - (1 / 2)
        right_limit = tick_locations[-1] + (1 / 2)

        ax.set_xlim(left_limit, right_limit)

        ax2 = ax.twiny()

        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(ax.get_xticks())
        synapse_labels = [format_synapses(s) for s in synapses]
        ax2.set_xticklabels(synapse_labels)#, fontsize=FONT_SCALE*9)
        plt.setp(ax.get_xticklabels(), rotation=15, ha='center', rotation_mode='anchor')
        ax2.tick_params(axis='x', pad=0)
        ax2.set_xlabel(r"Number of Synapses ($\times 10^{{12}}$)")

        
    fig.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved unified barplot visualization to {filename}")


def plot_construction_and_simulation_time(data, real_nodes, filename):
    """
    Creates a two-panel plot showing Network Construction and State Propagation times.
    Panel A (Left): Sum of construct and calibrate times.
    Panel B (Right): Simulation time.
    """
    # Create a 1x2 subplot figure. sharex=True links the x-axes.
    fig, axs = plt.subplots(1, 2, figsize=(16, 7), sharex=True)

    # panel labels
    panel_labels = ["a", "b"]

    data = convert_keys_to_int(data)

    for o in opt_levels_all:
        # prepare data for both panels
        color = colors[o+1]
        label = f"Opt {o+1} Sim." if o < 3 else "Opt 3 Sim.\nw/out spike recording"

        # panel A
        construct_avg = np.nan_to_num(np.array(data[o]["construct_avg"]))
        calibrate_avg = np.nan_to_num(np.array(data[o]["calibrate_avg"]))
        total_construction_avg = construct_avg + calibrate_avg

        construct_std = np.nan_to_num(np.array(data[o]["construct_std"]))
        calibrate_std = np.nan_to_num(np.array(data[o]["calibrate_std"]))
        total_construction_std = np.sqrt(construct_std**2 + calibrate_std**2)
        
        axs[0].errorbar(real_nodes, total_construction_avg, yerr=total_construction_std,
                        fmt='-o', capsize=5, color=color, label=label)

        # panel B
        simulate_avg = np.array(data[o]["simulate_avg"])
        simulate_std = np.array(data[o]["simulate_std"])
        
        axs[1].errorbar(real_nodes, simulate_avg, yerr=simulate_std,
                        fmt='-o', capsize=5, color=color, label=label)

    # -panel formatting
    axs[0].set_xlabel("Number of Nodes")
    axs[0].set_ylabel('Network construction [s]')
    if scale != 10:
        axs[0].set_ylim(bottom=7.6)
    axs[0].grid(alpha=0.8, linestyle=':')
    axs[0].text(-0.1, 1.0175, panel_labels[0], fontsize=25, weight="bold", color='k', transform=axs[0].transAxes)

    axs[1].set_ylabel(r'$T_{\mathrm{wall}} / T_{\mathrm{model}}$')
    axs[1].set_xlabel("Number of Nodes")
    axs[1].set_ylim(bottom=0)
    axs[1].grid(alpha=0.8, linestyle=':')
    axs[0].legend()
    axs[1].text(-0.1, 1.0175, panel_labels[1], fontsize=25, weight="bold", color='k', transform=axs[1].transAxes)

    for ax in axs:
        ax.set_xticks(real_nodes)
        ax.set_xlim(left=20, right=270)

    ax2 = axs[0].twiny()

    ax2.set_xlim(axs[0].get_xlim())
    ax2.set_xticks(axs[0].get_xticks())
    synapse_labels = [format_synapses(s) for s in synapses_real]
    ax2.set_xticklabels(synapse_labels)
    ax2.tick_params(axis='x', pad=0)
    ax2.set_xlabel(r"Number of Synapses ($\times 10^{{12}}$)")

    ax3 = axs[1].twiny()

    ax3.set_xlim(axs[1].get_xlim())
    ax3.set_xticks(axs[1].get_xticks())
    ax3.set_xticklabels(synapse_labels)
    ax3.tick_params(axis='x', pad=0)
    ax3.set_xlabel(r"Number of Synapses ($\times 10^{{12}}$)")

    fig.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    print(f"Saved {filename}")


def plot_gpu_memory_with_inset(data, real_nodes, fake_nodes, filename):
    """Plots GPU memory with a manually controlled inset zoom."""
    fig, ax = plt.subplots(figsize=(10, 7))

    all_real_peaks_min = []
    all_real_peaks_max = []

    GIB_TO_GB_FACTOR = 1.073741824

    data = convert_keys_to_int(data)

    for o in opt_levels_partial:
        real_peak = np.array(data[o]["gpu_mem_peak"])* GIB_TO_GB_FACTOR
        real_peak_std = np.array(data[o]["gpu_mem_peak_std"]) * GIB_TO_GB_FACTOR
        fake_peak = np.array(data[o]["fake_gpu_mem_peak"])* GIB_TO_GB_FACTOR
        
        # Keep track of the full range of real data for the zoom box
        all_real_peaks_min.append(np.nanmin(real_peak - real_peak_std))
        all_real_peaks_max.append(np.nanmax(real_peak + real_peak_std))
        
        color = colors[o+1]
        estimate_color_dark = darken_color(color)

        # plot on main axis
        ax.plot(fake_nodes, fake_peak, linestyle='--', marker='s', markersize=4, color=estimate_color_dark, label=f"Opt {o+1} Est.", zorder=1)
        ax.errorbar(real_nodes, real_peak, yerr=real_peak_std, fmt='-o', capsize=5, color=color, label=f"Opt {o+1} Sim.", zorder=2)

    # inset axis
    axins = ax.inset_axes([0.55, 0.15, 0.4, 0.35])
    
    for o in opt_levels_partial:
        real_peak = np.array(data[o]["gpu_mem_peak"]) * GIB_TO_GB_FACTOR
        real_peak_std = np.array(data[o]["gpu_mem_peak_std"]) * GIB_TO_GB_FACTOR
        fake_peak = np.array(data[o]["fake_gpu_mem_peak"]) * GIB_TO_GB_FACTOR
        color = colors[o+1]
        estimate_color_dark = darken_color(color)
        axins.plot(fake_nodes, fake_peak, linestyle='--', marker='s', markersize=4, color=estimate_color_dark)
        axins.errorbar(real_nodes, real_peak, yerr=real_peak_std, fmt='-o', markersize=4, capsize=3, color=color)

    x1, x2 = real_nodes[0]*0.9, real_nodes[-1]*1.1
    y1, y2 = np.nanmin(all_real_peaks_min)*0.975, np.nanmax(all_real_peaks_max)*1.025

    x_margin = (x2 - x1) * 0.05
    y_margin = (y2 - y1) * 0.05
    axins.set_xlim(x1 - x_margin, x2 + x_margin)
    axins.set_ylim(y1 - y_margin, y2 + y_margin)
    
    axins.set_facecolor('w')
    
    axins.grid(True, linestyle=':', alpha=0.6)
    axins.tick_params(axis='x', labelsize=12 * FONT_SCALE)
    axins.tick_params(axis='y', labelsize=12 * FONT_SCALE)
    axins.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
    axins.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
    axins.set_xticks([32, 64, 96, 128, 192, 256])

    ax2 = axins.twiny()

    ax2.set_xlim(axins.get_xlim())
    ax2.set_xticks(axins.get_xticks())
    synapse_labels = [format_synapses(s) for s in synapses_real]
    ax2.set_xticklabels(synapse_labels, fontsize=12 * FONT_SCALE)
    ax2.tick_params(axis='x', pad=0)


    rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                     fill=False, edgecolor="black", linestyle='-', alpha=0.7, lw=1)
    ax.add_patch(rect)

    p1 = ConnectionPatch(xyA=(x2, y2), xyB=(x1 - x_margin, y2 + y_margin),
                         coordsA=ax.transData, coordsB=axins.transData,
                         color="silver", lw=1)

    p2 = ConnectionPatch(xyA=(x2, y1), xyB=(x1 - x_margin, y1 - y_margin),
                         coordsA=ax.transData, coordsB=axins.transData,
                         color="silver", lw=1)
    ax.add_patch(p1)
    ax.add_patch(p2)

    ax.hlines(64.0, 0, max(fake_nodes)+1000, color='k', linestyles='-.')
    ax.text(0.035, 0.915, 'A100 GPU memory', fontsize=11 * FONT_SCALE, color='k', transform=ax.transAxes)

    # labels and titles
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("GPU Memory Peak [GB]")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.8, linestyle=':')
    if scale != 30:
        ax.set_xlim(left=0, right=4300)
    else:
        ax.set_xlim(left=0, right=1100)
    ax.legend(loc='lower left', ncol=2)

    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 64])
    if scale != 30:
        ax.set_xticks([512, 1024, 1536, 2048, 2560, 3072, 3584, 4096])
    else:
        ax.set_xticks([512, 1024])
    ax3 = ax.twiny()

    ax3.set_xlim(ax.get_xlim())
    ax3.set_xticks(ax.get_xticks())

    synapse_labels = [format_synapses(s) for s in synapses[6:]]
    ax3.set_xticklabels(synapse_labels)
    ax3.tick_params(axis='x', pad=0)
    ax3.set_xlabel(r"Number of Synapses ($\times 10^{{12}}$)")

    fig.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    print(f"Saved {filename}")


# main
if __name__ == "__main__":
    print("Loading and aggregating data...")
    try:
        
        # please unpack the data if you want to run these functions
        #data = load_and_aggregate_data(results_dir=f"./results_scale_{scale}/")
        #df_boxplot = prepare_data_for_barplot(data)

        # the output of the functions is provided here for convenience
        with open(f'results_scale_{scale}/aggregated_data_scale{scale}.json', 'r') as file:
            data = json.load(file)

        df_boxplot = pd.read_csv(f"results_scale_{scale}/barplot_data_scale{scale}.csv")
        
        print("Data loaded successfully.")

        # generate plots
        plot_unified_time_as_barplot(df_boxplot, "results_scale_{}/cond_construction_preparation_scale{}.pdf".format(scale, scale))
        plot_construction_and_simulation_time(data, real_nodes, "results_scale_{}/simulate_time{}.pdf".format(scale, scale))
        plot_gpu_memory_with_inset(data, real_nodes, fake_nodes, "results_scale_{}/gpu_memory_peak{}.pdf".format(scale, scale))

        print("\nPlots have been generated and saved as PDF files.")
        plt.show()
    except (FileNotFoundError) as e:
            print(f"\nERROR: {e}")
            print("Please ensure the script is run in the correct directory containing the data.")
