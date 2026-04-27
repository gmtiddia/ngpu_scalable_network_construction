#! /usr/bin/python3

import json
import re
import sys
import logging

from pathlib import Path
from argparse import ArgumentParser

import pandas
import seaborn as sns
import matplotlib.pyplot as plt


FONT_SCALE = 1.3
matplotlib_params = {
    "image.origin": "lower",
    "image.interpolation": "nearest",
    "axes.grid": False,
    "axes.labelsize": 15 * FONT_SCALE,
    "axes.titlesize": 19 * FONT_SCALE,
    "font.size": 16 * FONT_SCALE,
    "legend.fontsize": 11 * FONT_SCALE,
    "xtick.labelsize": 13 * FONT_SCALE,
    "ytick.labelsize": 13 * FONT_SCALE,
    "text.usetex": False,
}

LOG = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="./change_indegree/")
    parser.add_argument("--output_prefix", type=str, default="indegree_scaling")
    parser.add_argument("--output_type", type=str, default="pdf")
    parser.add_argument("--verbosity", type=int, default=20)
    args = parser.parse_args()

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(logging.Formatter("%(levelname)s:\n%(message)s"))
    logging.basicConfig(level=args.verbosity, handlers=(stdout,))

    sns.color_palette(palette="Set2")
    plt.rcParams.update(matplotlib_params)

    path = Path(args.path)
    if not path.is_dir():
        raise RuntimeError("Invalid input path")

    node_set = set()
    netcon_data_table = []
    calibrate_data_table = []
    pattern = re.compile(r".*indegree_x(\d+)/nodes_(\d+)(_fake)?/.*/log_(\d+).json")
    for fp in path.glob("**/*.json"):
        match: re.Match | None = pattern.match(str(fp))
        assert match is not None
        indegree_scale = int(match.group(1))
        nodes = int(match.group(2))
        run = "Estimated" if match.group(3) == "_fake" else "Simulated"
        rank = int(match.group(4))
        loaded_data = json.loads(fp.read_text())
        netcon_data_table.append(
            (
                run,
                indegree_scale,
                nodes,
                nodes * 4,
                rank,
                loaded_data["timers"]["time_construct"] / 1e9,
            )
        )
        calibrate_data_table.append(
            (
                run,
                indegree_scale,
                nodes,
                nodes * 4,
                rank,
                loaded_data["timers"]["time_calibrate"] / 1e9,
            )
        )
        node_set.add(nodes)

    netcon_df = pandas.DataFrame(
        netcon_data_table,
        columns=["Run type", "In-degree scale", "num_nodes", "num_gpus", "rank", "time"],
    ).sort_values("Run type", ascending=False)

    calibrate_df = pandas.DataFrame(
        calibrate_data_table,
        columns=["Run type", "In-degree scale", "num_nodes", "num_gpus", "rank", "time"],
    ).sort_values("Run type", ascending=False)

    LOG.info(netcon_df.head)
    LOG.info(calibrate_df.head)

    mosaic = (("ncs", "nce"), ("cs", "ce"))
    fig, axd = plt.subplot_mosaic(
        mosaic,  # pyright: ignore
        constrained_layout=True,
        figsize=(16, 18),
    )

    sns.lineplot(
        netcon_df[netcon_df["num_nodes"] <= 256],  # pyright: ignore
        x="num_gpus",
        y="time",
        style="Run type",
        hue="In-degree scale",
        err_style="bars",
        ax=axd["ncs"],
        legend=False,
    )

    sns.lineplot(
        calibrate_df[calibrate_df["num_nodes"] <= 256],  # pyright: ignore
        x="num_gpus",
        y="time",
        style="Run type",
        hue="In-degree scale",
        err_style="bars",
        ax=axd["cs"],
        legend=False,
    )

    sns.lineplot(
        netcon_df,
        x="num_gpus",
        y="time",
        style="Run type",
        hue="In-degree scale",
        err_style="bars",
        ax=axd["nce"],
    )

    sns.lineplot(
        calibrate_df,
        x="num_gpus",
        y="time",
        style="Run type",
        hue="In-degree scale",
        err_style="bars",
        ax=axd["ce"],
        legend=False,
    )

    all_nodes_data = sorted(node_set)
    all_gpus_data = [n * 4 for n in all_nodes_data]

    nodes_est = [96, 512, 1024, 2048, 3072, 4096]
    gpus_est = [n * 4 for n in nodes_est]

    for key, ax in axd.items():
        current_xlim = ax.get_xlim()
        gpus_ticks_in_view = [
            g for g in all_gpus_data if current_xlim[0] <= g <= current_xlim[1]
        ]
        nodes_labels_in_view = [
            n
            for n, g in zip(all_nodes_data, all_gpus_data)
            if current_xlim[0] <= g <= current_xlim[1]
        ]

        if "ncs" in key:
            ax.set_ylabel("Neuron and device\ncreation and connection [s]")
        elif "cs" in key:
            ax.set_ylabel("Simulation preparation time [s]")
        else:
            gpus_ticks_in_view = [g for g in gpus_est]
            nodes_labels_in_view = [
                n
                for n, g in zip(nodes_est, gpus_est)
                if current_xlim[0] <= g <= current_xlim[1]
            ]
            ax.set_ylabel("")

        if "nce" in key:
            plt.setp(ax.get_legend().get_texts(), fontsize=13 * FONT_SCALE)
            plt.setp(ax.get_legend().get_title(), fontsize=15 * FONT_SCALE)

        ax.grid(True, linestyle=":", alpha=0.7)

        ax.set_xlabel("Number of GPUs", labelpad=5)

        ax.set_xticks(gpus_ticks_in_view)

        secax = ax.twiny()
        secax.set_xlim(ax.get_xlim())

        secax.set_xticks(gpus_ticks_in_view)
        secax.tick_params(labelsize=13 * FONT_SCALE)
        secax.set_xticklabels(nodes_labels_in_view, fontsize=13 * FONT_SCALE)
        secax.set_xlabel("Number of nodes", fontsize=15 * FONT_SCALE)

        secax.spines["bottom"].set_position(("outward", 55))
        secax.spines["top"].set_visible(False)
        secax.xaxis.set_ticks_position("bottom")
        secax.xaxis.set_label_position("bottom")

    output_filename = f"{args.output_prefix}.{args.output_type}"
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    LOG.info(f"Figure saved to {output_filename}")

    plt.show()
