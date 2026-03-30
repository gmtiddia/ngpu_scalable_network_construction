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

LOG = logging.getLogger(__name__)


def update_verbosity(lvl: int) -> None:
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(logging.Formatter("%(levelname)s:\n%(message)s"))
    logging.basicConfig(level=lvl, handlers=(stdout,))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="golosio/change_indegree/")
    parser.add_argument("--output_prefix", type=str, default="indegree_scaling")
    parser.add_argument("--output_type", type=str, default="pdf")
    parser.add_argument("--verbosity", type=int, default=20)
    args = parser.parse_args()

    update_verbosity(args.verbosity)

    path = Path(args.path)
    if not path.is_dir():
        raise RuntimeError("Invalid input path")

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
                rank,
                loaded_data["timers"]["time_construct"] / 1e9,
            )
        )
        calibrate_data_table.append(
            (
                run,
                indegree_scale,
                nodes,
                rank,
                loaded_data["timers"]["time_calibrate"] / 1e9,
            )
        )

    netcon_df = pandas.DataFrame(
        netcon_data_table,
        columns=["run_type", "indegree_scale", "num_nodes", "rank", "time"],
    ).sort_values("run_type")

    calibrate_df = pandas.DataFrame(
        calibrate_data_table,
        columns=["run_type", "indegree_scale", "num_nodes", "rank", "time"],
    ).sort_values("run_type")

    LOG.info(netcon_df.head)
    LOG.info(calibrate_df.head)

    mosaic = (("ncs", "nce"), ("cs", "ce"))
    fig, axd = plt.subplot_mosaic(
        mosaic,  # pyright: ignore
        constrained_layout=True,
        # height_ratios=[0.5] * 2 + [1] * 9,
        # width_ratios=[1] * 2,
        figsize=(10, 16),
    )

    sns.set_theme(context="paper", style="ticks")

    sns.lineplot(
        netcon_df[netcon_df["num_nodes"] <= 256],  # pyright: ignore
        x="num_nodes",
        y="time",
        style="run_type",
        hue="indegree_scale",
        err_style="bars",
        ax=axd["ncs"],
    )

    sns.lineplot(
        calibrate_df[calibrate_df["num_nodes"] <= 256],  # pyright: ignore
        x="num_nodes",
        y="time",
        style="run_type",
        hue="indegree_scale",
        err_style="bars",
        ax=axd["cs"],
    )

    sns.lineplot(
        netcon_df,
        x="num_nodes",
        y="time",
        style="run_type",
        hue="indegree_scale",
        err_style="bars",
        ax=axd["nce"],
    )

    sns.lineplot(
        calibrate_df,
        x="num_nodes",
        y="time",
        style="run_type",
        hue="indegree_scale",
        err_style="bars",
        ax=axd["ce"],
    )

    plt.show()
