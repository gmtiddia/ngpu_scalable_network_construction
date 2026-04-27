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

tol_vibrant = [tc.light.orange, tc.vibrant.cyan, tc.pale.pale_red, tc.light.mint, tc.pale.pale_green, tc.vibrant.red, tc.vibrant.teal, tc.light.pale_grey]
darker = ['#f06537', '#1395eb', '#faa2a2', tc.bright.green, tc.pale.pale_green, tc.vibrant.red, tc.vibrant.teal, tc.light.pale_grey]
tol_vibrant_rgb = [mcolors.to_rgb(c) for c in tol_vibrant]
color_map = dict(zip(tol_vibrant_rgb, darker))
def darken_color(rgb_color):
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
scale = args.scale
allowed_scales = [10, 20, 30]
if scale not in allowed_scales:
    raise ValueError(f"invalid scale parameter: {scale}, allowed values are {', '.join(map(str, allowed_scales))}")
real_nodes = [32, 64, 96, 128, 192, 256]
if scale != 30:
    fake_nodes = [32, 64, 96, 128, 192, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]
else:
    fake_nodes = [32, 64, 96, 128, 192, 256, 512, 1024]
opt_levels_all = [-1, 0, 1, 2, 3]
opt_levels_partial = [-1, 0, 1, 2]
run_seeds = list(range(5))
ranks_per_node = 4
n_neurons_per_proc = 11250
indegree = 11250
synapses = [n_neurons_per_proc*indegree*ranks_per_node*scale*n for n in fake_nodes]
synapses_real = [n_neurons_per_proc*indegree*ranks_per_node*scale*n for n in real_nodes]
def format_synapses(total_conns): return fr"{total_conns / 1e12:.1f}"
def safe_mean(arr): return np.mean(arr) if arr and not np.all(np.isnan(arr)) else np.nan
def safe_std(arr): return np.std(arr) if arr and not np.all(np.isnan(arr)) else np.nan
def convert_keys_to_int(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            try: new_key = int(k)
            except (ValueError, TypeError): new_key = k
            new_dict[new_key] = convert_keys_to_int(v)
        return new_dict
    elif isinstance(obj, list): return [convert_keys_to_int(i) for i in obj]
    else: return obj

def load_and_aggregate_data(results_dir:str='./'):
    """Loads to a dictionary the data from the different simulations performed and saves the data to a json file.

    Args:
        results_dir (str, optional): directory in which results are stored. Defaults to './'.

    Returns:
        dict: Python dictionary with the data collected from each simulation folder.
    """
    data = {}
    for o in opt_levels_all:
        data[o] = {
            "construct_raw": [], "calibrate_raw": [], "simulate_raw": [], "gpu_mem_peak_raw": [],
            "fake_construct_raw": [], "fake_calibrate_raw": [], "fake_gpu_mem_peak_raw": [],
            "n_neurons": [], "n_connections": []
        }
        if o == 0:
            for N in fake_nodes:
                data[o]["n_neurons"].append(scale * ranks_per_node * N * n_neurons_per_proc)
                data[o]["n_connections"].append(scale * ranks_per_node * N * n_neurons_per_proc * indegree)
        
        for N in real_nodes:
            all_construct_vals, all_calibrate_vals, all_simulate_vals, all_gpu_peak_vals = [], [], [], []
            
            for r in run_seeds:
                folder = results_dir + f"nodes_{N}/opt_{o}_run_{r}"
                for rank in range(ranks_per_node * N):
                    file = os.path.join(folder, f"log_{rank}.json")
                    if os.path.exists(file):
                        with open(file, "r") as f: log = json.load(f)
                        t, g = log.get("timers", {}), log.get("gpu_mem", {})
                        all_construct_vals.append(t.get("time_construct", 0) / 1e9)
                        all_calibrate_vals.append(t.get("time_calibrate", 0) / 1e9)
                        all_simulate_vals.append(t.get("time_simulate", 0) / 1e9)
                        all_gpu_peak_vals.append(g.get("gpu_mem_peak", 0) / (2**30))
            
            data[o]["construct_raw"].append(all_construct_vals or [np.nan])
            data[o]["calibrate_raw"].append(all_calibrate_vals or [np.nan])
            data[o]["simulate_raw"].append(all_simulate_vals or [np.nan])
            data[o]["gpu_mem_peak_raw"].append(all_gpu_peak_vals or [np.nan])
            
        for N in fake_nodes:
            folder = results_dir + f"nodes_{N}_fake/opt_{o}_run_0"
            construct_vals, calibrate_vals, gpu_peak_vals = [], [], []
            for rank in range(4):
                file = os.path.join(folder, f"log_{rank}.json")
                if os.path.exists(file):
                    with open(file, "r") as f: log = json.load(f)
                    t, g = log.get("timers", {}), log.get("gpu_mem", {})
                    construct_vals.append(t.get("time_construct", 0) / 1e9)
                    calibrate_vals.append(t.get("time_calibrate", 0) / 1e9)
                    gpu_peak_vals.append(g.get("gpu_mem_peak", 0) / (2**30))
            
            data[o]["fake_construct_raw"].append(construct_vals or [np.nan])
            data[o]["fake_calibrate_raw"].append(calibrate_vals or [np.nan])
            data[o]["fake_gpu_mem_peak_raw"].append(gpu_peak_vals or [np.nan])
            
        with open('results_scale_{}/aggregated_data_scale{}.json'.format(int(scale), int(scale)), 'w') as fp:
            json.dump(data, fp, indent=4)
    return data


def prepare_data_for_barplot(data_agg):
    """Given the data loaded, returns a pandas DataFrame for plotting.

    Args:
        data_agg (dict): output of load_and_aggregate_data function

    Returns:
        pandas.DataFrame: DataFrame with the timers correctly handled for plotting.
    """
    plot_data = []
    quantities = ["Construct", "Calibrate"]
    data_agg = convert_keys_to_int(data_agg)
    for o in opt_levels_partial:
        for quantity_name in quantities:
            quantity_key = quantity_name.lower()
            
            per_node_raw_data = data_agg[o][f"{quantity_key}_raw"]
            for i, all_points_for_node in enumerate(per_node_raw_data):
                N = real_nodes[i]
                for point in all_points_for_node:
                    plot_data.append({"Nodes": N, "Time (s)": point, "Quantity": quantity_name, "Type": "Real", "Hue": f"Opt {o+1}"})
            
            per_node_fake_raw = data_agg[o][f"fake_{quantity_key}_raw"]
            for i, fake_points_for_node in enumerate(per_node_fake_raw):
                N_fake = fake_nodes[i]
                for point in fake_points_for_node:
                    plot_data.append({"Nodes": N_fake, "Time (s)": point, "Quantity": quantity_name, "Type": "Estimate", "Hue": f"Opt {o+1} Estimate"})

    df = pd.DataFrame(plot_data)
    df["GPUs"] = df["Nodes"]*4
    df.dropna(subset=['Time (s)'], inplace=True)
    df.to_csv(f"results_scale_{scale}/barplot_data_scale{scale}.csv")
    return df


def plot_unified_time_as_barplot(df_unified, filename):
    fig, axs = plt.subplots(2, 1, figsize=(10, 15), sharey=False)
    quantities = ["Construct", "Calibrate"]
    quantities_labels = {"Construct": "Neuron and device\ncreation and connection", "Calibrate": "Simulation\npreparation"}
    bar_palette = {f"Opt {o+1} Estimate": darken_color(colors[o+1]) for o in opt_levels_partial}
    estimate_hues = [f"Opt {o+1} Estimate" for o in opt_levels_partial]
    panel_label = ["a", "b"]

    def gpus_to_nodes(gpus_val):
        return gpus_val / 4
    def nodes_to_gpus(nodes_val):
        return nodes_val * 4

    for col, quantity_name in enumerate(quantities):
        ax = axs[col]; df_subset = df_unified[df_unified['Quantity'] == quantity_name]; df_real = df_subset[df_subset['Type'] == 'Real']; df_estimate = df_subset[df_subset['Type'] == 'Estimate']
        
        sns.barplot(
            data=df_estimate, x="GPUs", y="Time (s)", hue="Hue", hue_order=estimate_hues,
            palette=bar_palette, ax=ax, edgecolor='black', linewidth=0.0, legend=False,
            errorbar='sd' # standard deviation for errorbar
        )

        stats_real = df_real.groupby(['GPUs', 'Hue'])['Time (s)'].agg(['mean', 'std']).reset_index()
        
        nodes_unique = df_estimate['GPUs'].unique(); node_map = {node: i for i, node in enumerate(nodes_unique)}; 
        n_hues = len(estimate_hues)
        group_width = 0.8
        bar_width = group_width / n_hues
        
        for i, hue_est in enumerate(estimate_hues):
            offset = (i - (n_hues - 1) / 2) * bar_width
            hue_real_name = hue_est.replace(' Estimate', '')
            data_to_plot = stats_real[stats_real['Hue'] == hue_real_name]
            
            if not data_to_plot.empty:
                x_pos_real = data_to_plot['GPUs'].map(node_map).values + offset
                ax.errorbar(x_pos_real, data_to_plot['mean'], yerr=data_to_plot['std'],
                            fmt='_', color='brown', markersize=5, capsize=3.5,
                            elinewidth=1.2, markeredgewidth=1, linestyle='none')
        
        ax.text(-0.1, 1.0175, panel_label[col], fontsize=25, weight="bold", color='k', transform=ax.transAxes); ax.set_xlabel("Number of GPUs"); ax.set_ylabel(f"{quantities_labels[quantity_name]} [s]")
        if col == 0:
            from matplotlib.lines import Line2D; from matplotlib.patches import Patch
            handles = [Patch(facecolor=darken_color(colors[o+1]), edgecolor='black', linewidth=0., label=f"GML {o+1} Est.") for o in opt_levels_partial]
            
            tmp_fig, tmp_ax = plt.subplots()
            eb = tmp_ax.errorbar(
                [0], [0], yerr=[1],
                fmt='_', color='brown',
                markersize=5, capsize=3.5,
                elinewidth=1.2, markeredgewidth=1.5,
                label= "Sim. results"
            )
            plt.close(tmp_fig)
            
            handles.append(eb); ax.legend(handles=handles, ncol=1)

        tick_locations = ax.get_xticks()

        for tick in tick_locations[::2]: ax.axvspan(tick - 0.5, tick + 0.5, color='lightgray', alpha=0.5, zorder=0)
        ax.set_xlim(tick_locations[0] - 0.5, tick_locations[-1] + 0.5)
        ax2 = ax.twiny(); ax2.set_xlim(ax.get_xlim()); ax2.set_xticks(ax.get_xticks()); synapse_labels = [format_synapses(s) for s in synapses]; ax2.set_xticklabels(synapse_labels); plt.setp(ax.get_xticklabels(), rotation=20, y=-0.005, ha='center', rotation_mode='anchor'); ax2.tick_params(axis='x', pad=0); ax2.set_xlabel(r"Number of synapses ($\times 10^{{12}}$)")

    secax = ax.secondary_xaxis(-0.17, functions=(gpus_to_nodes, nodes_to_gpus))
    secax.set_xlabel("Number of nodes")
    secax.set_xticks(np.asarray(tick_locations)/4)
    gpu_ticks = [int(t.get_text()) for t in ax.get_xticklabels()]
    node_labels = [int(gpus_to_nodes(g)) for g in gpu_ticks]

    secax.set_xticklabels(node_labels, rotation=15)

    fig.tight_layout(); plt.savefig(filename, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved unified barplot visualization to {filename}")


def plot_construction_and_simulation_time(data, real_nodes, filename):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharex=True) # Ho aumentato un po' l'altezza per fare spazio
    panel_labels = ["a", "b"]
    data = convert_keys_to_int(data)
    
    # I valori per l'asse X principale sono il numero di GPU
    gpus = np.asarray(real_nodes) * 4
    
    # --- PLOTTING DEI DATI (questa parte è corretta) ---
    for o in opt_levels_all:
        color, label = colors[o+1], f"GML {o+1} Sim." if o < 3 else "GML 3 Sim.\nw/out spike recording"
        
        # Subplot 0: Tempo di costruzione
        construct_raw = data[o]["construct_raw"]; calibrate_raw = data[o]["calibrate_raw"]
        construct_avg = np.array([safe_mean(raw) for raw in construct_raw])
        calibrate_avg = np.array([safe_mean(raw) for raw in calibrate_raw])
        total_construction_avg = np.nan_to_num(construct_avg + calibrate_avg)
        construct_std = np.array([safe_std(raw) for raw in construct_raw])
        calibrate_std = np.array([safe_std(raw) for raw in calibrate_raw])
        total_construction_std = np.sqrt(construct_std**2 + calibrate_std**2)
        axs[0].errorbar(gpus, total_construction_avg, yerr=total_construction_std, fmt='-o', capsize=5, color=color, label=label)
        
        # Subplot 1: Tempo di simulazione
        simulate_raw = data[o]["simulate_raw"]
        simulate_avg = np.array([safe_mean(raw) for raw in simulate_raw])
        simulate_std = np.array([safe_std(raw) for raw in simulate_raw])
        axs[1].errorbar(gpus, simulate_avg, yerr=simulate_std, fmt='-o', capsize=5, color=color, label=label)


    # 1. Definiamo le funzioni per convertire tra GPU e Nodi
    def gpus_to_nodes(gpus_val):
        return gpus_val / 4
    def nodes_to_gpus(nodes_val):
        return nodes_val * 4

    # 2. Applichiamo le impostazioni a entrambi i subplot (axs[0] e axs[1])
    for i, ax in enumerate(axs):
        # Impostazioni per l'asse X primario (quello più in alto, "Number of GPUs")
        ax.set_xlabel("Number of GPUs")
        ax.set_xticks(gpus) # I tick devono corrispondere ai dati plottati!
        ax.set_xlim(left=120, right=1050)
        ax.grid(alpha=0.8, linestyle=':')
        ax.text(-0.1, 1.0175, panel_labels[i], fontsize=25, weight="bold", color='k', transform=ax.transAxes)

        # Creazione dell'asse X secondario (quello sotto, "Number of nodes")
        # Il parametro -0.25 lo sposta in basso (fuori dall'area del grafico)
        secax = ax.secondary_xaxis(-0.15, functions=(gpus_to_nodes, nodes_to_gpus))
        secax.set_xlabel("Number of nodes")
        secax.set_xticks(real_nodes) # Usiamo i valori dei nodi per i tick
        secax.set_xticklabels(real_nodes) # E come etichette

    # Impostazioni specifiche dei singoli subplot
    axs[0].set_ylabel('Network construction [s]')
    if scale != 10: axs[0].set_ylim(bottom=7.6)
    axs[0].legend()
    
    axs[1].set_ylabel(r'$T_{\mathrm{wall}} / T_{\mathrm{model}}$')
    # L'xlabel di axs[1] è già stato impostato a "Number of GPUs" nel ciclo
    axs[1].set_ylim(bottom=0)

    # Assi superiori per i sinapsi (twiny) - questo codice è corretto
    synapse_labels = [format_synapses(s) for s in synapses_real]
    for ax in axs:
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(ax.get_xticks())
        ax_top.set_xticklabels(synapse_labels)
        ax_top.tick_params(axis='x', pad=0)
        ax_top.set_xlabel(r"Number of synapses ($\times 10^{{12}}$)")

    fig.tight_layout()
    # Aggiungiamo un aggiustamento per dare spazio all'etichetta dell'asse inferiore
    plt.subplots_adjust(bottom=0.2)
    
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    print(f"Saved {filename}")


def plot_gpu_memory_with_inset(data, real_nodes, fake_nodes, filename):
    fig, ax = plt.subplots(figsize=(10, 7))
    all_real_peaks_min, all_real_peaks_max = [], []
    GIB_TO_GB_FACTOR = 1.073741824
    data = convert_keys_to_int(data)

    real_nodes = np.asarray(real_nodes)
    fake_nodes = np.asarray(fake_nodes)

    def gpus_to_nodes(gpus_val):
        return gpus_val / 4
    def nodes_to_gpus(nodes_val):
        return nodes_val * 4


    for o in opt_levels_partial:
        real_peak_raw = data[o]["gpu_mem_peak_raw"]
        real_peak_mean = np.array([safe_mean(raw) for raw in real_peak_raw]) * GIB_TO_GB_FACTOR
        real_peak_std = np.array([safe_std(raw) for raw in real_peak_raw]) * GIB_TO_GB_FACTOR
        
        fake_peak_raw = data[o]["fake_gpu_mem_peak_raw"]
        fake_peak_mean = np.array([safe_mean(raw) for raw in fake_peak_raw]) * GIB_TO_GB_FACTOR
        
        all_real_peaks_min.append(np.nanmin(real_peak_mean - real_peak_std)); all_real_peaks_max.append(np.nanmax(real_peak_mean + real_peak_std)); color, estimate_color_dark = colors[o+1], darken_color(colors[o+1]);
        ax.plot(nodes_to_gpus(fake_nodes), fake_peak_mean, linestyle='--', marker='s', markersize=4, color=estimate_color_dark, label=f"GML {o+1} Est.", zorder=1)
        ax.errorbar(nodes_to_gpus(real_nodes), real_peak_mean, yerr=real_peak_std, fmt='-o', capsize=5, color=color, label=f"GML {o+1} Sim.", zorder=2)
    
    axins = ax.inset_axes([0.55, 0.15, 0.4, 0.35])
    for o in opt_levels_partial:
        real_peak_raw = data[o]["gpu_mem_peak_raw"]; real_peak_mean = np.array([safe_mean(raw) for raw in real_peak_raw]) * GIB_TO_GB_FACTOR; real_peak_std = np.array([safe_std(raw) for raw in real_peak_raw]) * GIB_TO_GB_FACTOR
        fake_peak_raw = data[o]["fake_gpu_mem_peak_raw"]; fake_peak_mean = np.array([safe_mean(raw) for raw in fake_peak_raw]) * GIB_TO_GB_FACTOR
        color, estimate_color_dark = colors[o+1], darken_color(colors[o+1])
        axins.plot(nodes_to_gpus(fake_nodes), fake_peak_mean, linestyle='--', marker='s', markersize=4, color=estimate_color_dark)
        axins.errorbar(nodes_to_gpus(real_nodes), real_peak_mean, yerr=real_peak_std, fmt='-o', markersize=4, capsize=3, color=color)
    x1, x2 = nodes_to_gpus(real_nodes[0])*0.9, nodes_to_gpus(real_nodes[-1])*1.1
    y1, y2 = np.nanmin(all_real_peaks_min)*0.975, np.nanmax(all_real_peaks_max)*1.025
    x_margin, y_margin = (x2-x1)*0.05, (y2-y1)*0.05; axins.set_xlim(x1 - x_margin, x2 + x_margin)
    axins.set_ylim(y1 - y_margin, y2 + y_margin)
    axins.set_facecolor('w')
    axins.grid(True, linestyle=':', alpha=0.6)
    axins.tick_params(axis='x', labelsize=10 * FONT_SCALE)
    axins.tick_params(axis='y', labelsize=10 * FONT_SCALE)
    axins.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
    axins.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
    axins.set_xticks(nodes_to_gpus(np.asarray([32, 64, 96, 128, 192, 256])))


    ax2 = axins.twiny()
    ax2.set_xlim(axins.get_xlim())
    ax2.set_xticks(axins.get_xticks())
    synapse_labels = [format_synapses(s) for s in synapses_real]
    ax2.set_xticklabels(synapse_labels, fontsize=12 * FONT_SCALE)
    ax2.tick_params(axis='x', pad=0)
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="black", linestyle='-', alpha=0.7, lw=1); ax.add_patch(rect)
    p1 = ConnectionPatch(xyA=(x2, y2), xyB=(x1-x_margin, y2+y_margin), coordsA=ax.transData, coordsB=axins.transData, color="silver", lw=1); p2 = ConnectionPatch(xyA=(x2, y1), xyB=(x1-x_margin, y1-y_margin), coordsA=ax.transData, coordsB=axins.transData, color="silver", lw=1); ax.add_patch(p1); ax.add_patch(p2)
    ax.hlines(64.0, 0, nodes_to_gpus(max(fake_nodes))+1000, color='k', linestyles='-.'); ax.text(0.035, 0.915, 'A100 GPU memory', fontsize=11*FONT_SCALE, color='k', transform=ax.transAxes)
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("GPU memory peak [GB]")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.8, linestyle=':')

    

    if scale != 30: 
        ax.set_xlim(left=0, right=16500)
        nodes = np.asarray([512, 1024, 1536, 2048, 2560, 3072, 3584, 4096])
        ax.set_xticks(nodes_to_gpus(nodes))
        
    else: 
        ax.set_xlim(left=0, right=4150)
        nodes = np.asarray([512, 1024])
        ax.set_xticks(nodes_to_gpus(nodes))
    
    secax = ax.secondary_xaxis(-0.16, functions=(gpus_to_nodes, nodes_to_gpus))
    secax.set_xlabel("Number of nodes")
    secax.set_xticks(nodes)
    secax.set_xticklabels(nodes)

    ax.legend(loc='lower left', ncol=2); ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 64])
    ax3 = ax.twiny(); ax3.set_xlim(ax.get_xlim()); ax3.set_xticks(ax.get_xticks()); synapse_labels = [format_synapses(s) for s in synapses[6:]]; ax3.set_xticklabels(synapse_labels); ax3.tick_params(axis='x', pad=0); ax3.set_xlabel(r"Number of synapses ($\times 10^{{12}}$)")





    fig.tight_layout(); plt.savefig(filename, format="pdf", bbox_inches="tight"); print(f"Saved {filename}")


def plot_difference_with_fit(df_unified, filename):
    """
    Crea un plot a due pannelli:
    - Sinistra: Differenza Relativa (%) SENZA fit.
    - Destra: Differenza Assoluta (s) CON fit lineare.
    Entrambi i pannelli usano un doppio asse X (GPU e Nodi).
    """
    fig, axs = plt.subplots(1, 2, figsize=(18, 8), sharey=False)
    quantity_name = "Construct"
    panel_labels = ["a", "b"]

    # --- 1. PREPARAZIONE DEI DATI (invariata) ---
    df_opt0 = df_unified[df_unified['Hue'].str.contains("Opt 0")].copy()
    df_subset = df_opt0[df_opt0['Quantity'] == quantity_name]
    
    df_stats = df_subset.groupby(['Nodes', 'GPUs', 'Type'])['Time (s)'].agg(['mean', 'std']).unstack()
    df_stats.dropna(inplace=True)
    df_stats.reset_index(inplace=True)

    nodes_data = df_stats['Nodes'].values
    gpus_data = df_stats['GPUs'].values
    mean_real = df_stats[('mean', 'Real')].values
    mean_estimate = df_stats[('mean', 'Estimate')].values
    std_real = df_stats[('std', 'Real')].values
    std_estimate = df_stats[('std', 'Estimate')].values
    
    diff_abs = mean_real - mean_estimate
    error_abs = np.sqrt(std_real**2 + std_estimate**2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        diff_rel = (diff_abs / mean_real) * 100
        error_rel = np.abs(diff_rel) * np.sqrt(np.nan_to_num((error_abs / diff_abs)**2) + np.nan_to_num((std_real / mean_real)**2))
    
    plot_data = {
        0: {'y': diff_rel, 'yerr': error_rel, 'ylabel': 'Percentual difference [%]'},
        1: {'y': diff_abs, 'yerr': error_abs, 'ylabel': 'Difference (Sim. - Est.) [s]'}
    }

    # --- 2. PLOTTING CON CORREZIONI PER TICKS E RANGE ---
    # Trova il massimo dei dati misurati per impostare il limite dell'asse X
    max_gpus_data = max(gpus_data)

    for i, ax in enumerate(axs):
        data = plot_data[i]
        
        ax.errorbar(gpus_data, data['y'], yerr=data['yerr'],
                    fmt='o', color="firebrick",
                    label='Observed Difference')

        if i == 1:
            y_fit_data = data['y']
            coeffs = np.polyfit(nodes_data, y_fit_data, 1)
            poly_func = np.poly1d(coeffs)
            r_squared = 1 - (np.sum((y_fit_data - poly_func(nodes_data))**2) / np.sum((y_fit_data - np.mean(y_fit_data))**2))
            
            # La curva del fit può estendersi, ma verrà tagliata dal set_xlim
            nodes_curve = np.array([min(nodes_data), max(nodes_data)*1.1])
            gpus_curve = nodes_curve * 4
            y_curve = poly_func(nodes_curve)
            ax.plot(gpus_curve, y_curve, color=darker[0], linestyle='--', lw=2, label=f'Linear Fit (R² = {r_squared:.3f})')
        
        # --- Formattazione ---
        ax.set_ylabel(data['ylabel'])
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend()
        ax.text(-0.1, 1.05, panel_labels[i], fontsize=25, weight="bold", transform=ax.transAxes)
        
        # --- MODIFICA 1: Imposta il range basato sui dati misurati ---
        ax.set_xlim(left=0, right=max_gpus_data * 1.05)
        
        # --- MODIFICA 2 (CRUCIALE): Imposta i tick del primo asse ESATTAMENTE sui dati ---
        ax.set_xticks(gpus_data)
        ax.tick_params(axis='x')
        
    # --- 3. CREAZIONE DEL DOPPIO ASSE X (con logica corretta) ---
    for ax in axs:
        # L'etichetta dell'asse primario (GPU) va qui
        ax.set_xlabel("Number of GPUs", labelpad=0)

        # Crea l'asse secondario per i Nodi
        secax = ax.twiny()
        secax.set_xlim(ax.get_xlim()) # Allinea i limiti
        
        # --- MODIFICA 3 (CRUCIALE): Usa le stesse posizioni dati per i tick del secondo asse ---
        secax.set_xticks(gpus_data) 
        # ... ma usa le etichette dei nodi corrispondenti
        secax.set_xticklabels(nodes_data)
        
        secax.set_xlabel("Number of nodes")

        # Sposta fisicamente l'asse in basso (codice invariato)
        secax.spines['bottom'].set_position(('outward', 75))
        secax.spines['top'].set_visible(False)
        secax.xaxis.set_ticks_position('bottom')
        secax.xaxis.set_label_position('bottom')
        secax.tick_params(axis='x')
        
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    print(f"Saved difference plot with fit to {filename}")



if __name__ == "__main__":
    print("Aggregating all raw data points...")
    try:
        data = load_and_aggregate_data(results_dir=f"./results_scale_{scale}/")
        df_boxplot = prepare_data_for_barplot(data)
        print(df_boxplot)
        print("Data loaded and prepared successfully.")

        #plot_unified_time_as_barplot(df_boxplot, "results_scale_{}/cond_construction_preparation_scale{}.pdf".format(scale, scale))
        plot_construction_and_simulation_time(data, real_nodes, "results_scale_{}/simulate_time{}.pdf".format(scale, scale))
        #plot_gpu_memory_with_inset(data, real_nodes, fake_nodes, "results_scale_{}/gpu_memory_peak{}.pdf".format(scale, scale))
        #plot_difference_with_fit(df_boxplot, "results_scale_{}/difference_fit_scale{}.pdf".format(scale, scale))

        print("\nPlots have been generated and saved as PDF files.")
        plt.show()
    except (FileNotFoundError) as e:
        print(f"\nERROR: {e}")
        print("Please ensure the script is run in the correct directory containing the data.")


