#!/usr/bin/env python3
"""
Paper-ready analysis for Hybrid SFL fraud-detection benchmarks.

Run from the repository root.

Examples:
  python scripts/paper_analysis/paper_analysis.py --benchmark all --clean
  python scripts/paper_analysis/paper_analysis.py --benchmark main --clean
  python scripts/paper_analysis/paper_analysis.py --benchmark clients --clean --add-main-c4
  python scripts/paper_analysis/paper_analysis.py --benchmark stress --clean

Expected folders:
  outputs/main_benchmark/
  outputs/client_count_ablation/
  outputs/stress_non_iid/
  logs/creditcard/                         # optional GPU logs
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


RUN_RE = re.compile(
    r"(?P<protocol>[^_/]+)"
    r"_c(?P<num_clients>\d+)"
    r"_alpha(?P<alpha>[\d.]+)"
    r"_seed(?P<seed>\d+)"
    r"_r(?P<rounds>\d+)"
    r"(?:_split(?P<split_point>[^_]+))?"
    r"(?:_run(?P<run_id>\d+))?$"
)

DETECTION_COLS = [
    "eval_auprc", "eval_auroc", "eval_f1", "eval_precision", "eval_recall",
    "eval_bmr_risk", "eval_bmr_f1", "eval_bmr_precision", "eval_bmr_recall",
    "eval_brier_score", "eval_loss", "eval_accuracy",
]

SYSTEM_COLS = [
    "fit_round_time", "fit_elapsed_time", "fit_train_time",
    "fit_activation_comm_mb", "fit_cumulative_activation_comm_mb",
]

META_COLS = [
    "benchmark", "run_name", "protocol", "num_clients", "alpha",
    "seed", "rounds", "split_point", "run_id", "round",
]


def ensure_dir(path: Path, clean: bool = False) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_run_name(name: str, benchmark: str) -> Optional[dict]:
    m = RUN_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "benchmark": benchmark,
        "run_name": name,
        "protocol": d["protocol"],
        "num_clients": int(d["num_clients"]),
        "alpha": float(d["alpha"]),
        "seed": int(d["seed"]),
        "rounds": int(d["rounds"]),
        "split_point": d.get("split_point") or "unknown",
        "run_id": int(d["run_id"]) if d.get("run_id") else None,
    }


def find_run_dirs(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        return []
    direct = [p for p in input_dir.iterdir() if p.is_dir() and RUN_RE.match(p.name)]
    if direct:
        return sorted(direct)
    return sorted({p for p in input_dir.rglob("*") if p.is_dir() and RUN_RE.match(p.name)})


def load_yaml(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return None
    try:
        return yaml.safe_load(text)
    except Exception:
        return yaml.load(text, Loader=yaml.FullLoader)


def is_pair_list(value) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and isinstance(value[0], (list, tuple))
        and len(value[0]) >= 2
        and isinstance(value[0][0], (int, float))
    )


def to_value(x):
    try:
        return float(x)
    except Exception:
        return x


def yaml_to_rounds(data, prefix: str):
    out = {}

    if not isinstance(data, dict):
        return out

    for metric, value in data.items():
        col = f"{prefix}_{metric}"

        # Case 1:
        # metric:
        #   - [1, value]
        #   - [2, value]
        if is_pair_list(value):
            for pair in value:
                rnd = int(pair[0])
                out.setdefault(rnd, {})[col] = to_value(pair[1])

        # Case 2:
        # metric:
        #   - value_round_1
        #   - value_round_2
        #
        # This is the format your repo currently uses.
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                rnd = idx + 1
                out.setdefault(rnd, {})[col] = to_value(item)

        # Case 3:
        # metric:
        #   1: value
        #   2: value
        elif isinstance(value, dict):
            for k, v in value.items():
                try:
                    rnd = int(k)
                except Exception:
                    continue

                out.setdefault(rnd, {})[col] = to_value(v)

        # Case 4:
        # scalar metric
        else:
            out.setdefault(0, {})[col] = to_value(value)

    return out

def load_run(run_dir: Path, benchmark: str) -> Tuple[pd.DataFrame, dict]:
    meta = parse_run_name(run_dir.name, benchmark)
    if meta is None:
        return pd.DataFrame(), {"benchmark": benchmark, "run_name": run_dir.name, "status": "UNPARSEABLE"}

    fit_path = run_dir / "fit_metrics.yaml"
    eval_path = run_dir / "eval_metrics.yaml"
    fit = yaml_to_rounds(load_yaml(fit_path), "fit")
    ev = yaml_to_rounds(load_yaml(eval_path), "eval")

    merged: Dict[int, Dict[str, object]] = {}
    for d in [fit, ev]:
        for rnd, vals in d.items():
            merged.setdefault(rnd, {}).update(vals)

    status = dict(meta)
    status.update({
        "path": str(run_dir),
        "has_fit_metrics": fit_path.exists(),
        "has_eval_metrics": eval_path.exists(),
        "n_round_rows": len([r for r in merged if r != 0]),
    })

    rows = []
    for rnd, vals in sorted(merged.items()):
        if rnd == 0:
            continue
        row = dict(meta)
        row["round"] = rnd
        row.update(vals)
        rows.append(row)

    if not rows:
        status["status"] = "MISSING_METRICS"
        return pd.DataFrame(), status

    df = pd.DataFrame(rows).sort_values(["run_name", "round"])
    if "fit_activation_comm_mb" in df.columns:
        df["fit_cumulative_activation_comm_mb"] = (
            pd.to_numeric(df["fit_activation_comm_mb"], errors="coerce").fillna(0).cumsum()
        )
    status["status"] = "METRICS_FOUND"
    return df, status


def collect(input_dir: Path, benchmark: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts, statuses = [], []
    for run_dir in find_run_dirs(input_dir):
        df, st = load_run(run_dir, benchmark)
        statuses.append(st)
        if not df.empty:
            parts.append(df)

    round_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    status_df = pd.DataFrame(statuses)

    if round_df.empty:
        final_df = pd.DataFrame()
    else:
        final_df = (
            round_df.sort_values(["run_name", "round"])
            .groupby("run_name", as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )
    return round_df, final_df, status_df


def read_gpu_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(
            path,
            header=None,
            names=["timestamp", "gpu_name", "gpu_util_pct", "gpu_memory_mb", "gpu_power_w"],
        )
    except Exception:
        return None
    for c in ["gpu_util_pct", "gpu_memory_mb", "gpu_power_w"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def gpu_summary(final_df: pd.DataFrame, logs_dir: Optional[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if final_df.empty or logs_dir is None or not logs_dir.exists():
        return pd.DataFrame(), pd.DataFrame()

    process_rows, run_rows = [], []
    for _, row in final_df.iterrows():
        run_name = row["run_name"]
        run_id = row.get("run_id")
        d = logs_dir / run_name
        run_row = {"run_name": run_name, "run_id": run_id, "gpu_logs_found": False}
        if not d.exists():
            run_rows.append(run_row)
            continue

        local_rows = []
        for p in sorted(d.glob("gpu_*.csv")):
            df = read_gpu_csv(p)
            if df is None or df.empty:
                continue
            role = "server" if "server" in p.name else "client"
            m = re.search(r"client(\d+)", p.name)
            if m:
                role = f"client{m.group(1)}"
            r = {
                "run_name": run_name, "run_id": run_id, "role": role,
                "gpu_log_file": p.name, "gpu_sample_count": len(df),
                "gpu_util_mean_pct": df["gpu_util_pct"].mean(),
                "gpu_util_max_pct": df["gpu_util_pct"].max(),
                "gpu_memory_mean_mb": df["gpu_memory_mb"].mean(),
                "gpu_memory_max_mb": df["gpu_memory_mb"].max(),
                "gpu_power_mean_w": df["gpu_power_w"].mean(),
                "gpu_power_max_w": df["gpu_power_w"].max(),
            }
            local_rows.append(r)
            process_rows.append(r)

        if local_rows:
            tmp = pd.DataFrame(local_rows)
            clients = tmp[tmp["role"].str.startswith("client")]
            servers = tmp[tmp["role"].eq("server")]
            run_row.update({
                "gpu_logs_found": True,
                "gpu_process_count": len(tmp),
                "server_gpu_util_mean_pct": servers["gpu_util_mean_pct"].mean() if not servers.empty else np.nan,
                "server_gpu_util_max_pct": servers["gpu_util_max_pct"].max() if not servers.empty else np.nan,
                "server_gpu_memory_max_mb": servers["gpu_memory_max_mb"].max() if not servers.empty else np.nan,
                "client_gpu_util_mean_pct": clients["gpu_util_mean_pct"].mean() if not clients.empty else np.nan,
                "client_gpu_util_max_pct": clients["gpu_util_max_pct"].max() if not clients.empty else np.nan,
                "client_gpu_memory_max_mb": clients["gpu_memory_max_mb"].max() if not clients.empty else np.nan,
                "overall_gpu_util_max_pct": tmp["gpu_util_max_pct"].max(),
                "overall_gpu_memory_max_mb": tmp["gpu_memory_max_mb"].max(),
            })
        run_rows.append(run_row)

    proc_df = pd.DataFrame(process_rows)
    run_df = pd.DataFrame(run_rows)
    return proc_df, run_df


def add_gpu_and_efficiency(final_df: pd.DataFrame, logs_dir: Optional[Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    proc, run_gpu = gpu_summary(final_df, logs_dir)
    if not final_df.empty and not run_gpu.empty:
        final_df = final_df.merge(run_gpu.drop(columns=["run_id"], errors="ignore"), on="run_name", how="left")

    if not final_df.empty:
        if {"eval_auprc", "fit_elapsed_time"}.issubset(final_df.columns):
            final_df["auprc_per_second"] = final_df["eval_auprc"] / final_df["fit_elapsed_time"].replace(0, np.nan)
        if {"eval_f1", "fit_elapsed_time"}.issubset(final_df.columns):
            final_df["f1_per_second"] = final_df["eval_f1"] / final_df["fit_elapsed_time"].replace(0, np.nan)
        if {"eval_auprc", "fit_cumulative_activation_comm_mb"}.issubset(final_df.columns):
            final_df["auprc_per_comm_mb"] = final_df["eval_auprc"] / final_df["fit_cumulative_activation_comm_mb"].replace(0, np.nan)
        if {"eval_f1", "fit_cumulative_activation_comm_mb"}.issubset(final_df.columns):
            final_df["f1_per_comm_mb"] = final_df["eval_f1"] / final_df["fit_cumulative_activation_comm_mb"].replace(0, np.nan)

    return final_df, proc, run_gpu


def existing(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def mean_std_table(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    metric_cols = existing(df, metric_cols)
    if df.empty or not metric_cols:
        return pd.DataFrame()
    g = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "count"])
    g.columns = ["_".join([str(x) for x in c if x]) for c in g.columns]
    g = g.reset_index()
    first_count = f"{metric_cols[0]}_count"
    if first_count in g.columns:
        g["n_runs"] = g[first_count]
    return g


def fmt(mean, std, digits=4):
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def paper_table(summary: pd.DataFrame, group_cols: List[str], metrics: List[str], digits=4) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    out = summary[group_cols + (["n_runs"] if "n_runs" in summary.columns else [])].copy()
    for m in metrics:
        mc, sc = f"{m}_mean", f"{m}_std"
        if mc in summary.columns:
            out[m] = [fmt(a, b, digits) for a, b in zip(summary[mc], summary[sc] if sc in summary else [np.nan] * len(summary))]
    return out


def alpha_order(values):
    preferred = [100.0, 1.0, 0.5, 0.1]
    found = set(float(v) for v in values if pd.notna(v))
    return [x for x in preferred if x in found] + sorted([x for x in found if x not in preferred], reverse=True)


def alpha_label(x):
    x = float(x)
    if x == 100.0:
        return "100"
    if x.is_integer():
        return str(int(x))
    return str(x)


def plot_rounds(df, line_col, y_col, title, ylabel, out_path):
    if df.empty or not {"round", line_col, y_col}.issubset(df.columns):
        return
    plt.figure(figsize=(8, 5))
    for label, g in df.groupby(line_col):
        d = g[["round", y_col]].dropna().groupby("round", as_index=False)[y_col].mean().sort_values("round")
        if not d.empty:
            plt.plot(d["round"], d[y_col], marker="o", label=str(label))
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_category_lines(df, x_col, line_col, y_col, title, ylabel, out_path, xlabel=None, order=None):
    if df.empty or not {x_col, line_col, y_col}.issubset(df.columns):
        return
    d = df[[x_col, line_col, y_col]].dropna().groupby([x_col, line_col], as_index=False)[y_col].mean()
    if d.empty:
        return
    xs = order or sorted(d[x_col].unique())
    xpos = np.arange(len(xs))
    plt.figure(figsize=(8, 5))
    for label, g in d.groupby(line_col):
        ys = []
        for x in xs:
            hit = g[g[x_col].eq(x)]
            ys.append(hit[y_col].iloc[0] if not hit.empty else np.nan)
        plt.plot(xpos, ys, marker="o", label=str(label))
    plt.xticks(xpos, [str(x) for x in xs])
    plt.xlabel(xlabel or x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_grouped_bars(df, x_col, group_col, y_col, title, ylabel, out_path, xlabel=None, order=None):
    if df.empty or not {x_col, group_col, y_col}.issubset(df.columns):
        return
    d = df[[x_col, group_col, y_col]].dropna().groupby([x_col, group_col], as_index=False)[y_col].mean()
    if d.empty:
        return
    xs = order or sorted(d[x_col].unique())
    groups = list(d[group_col].unique())
    xpos = np.arange(len(xs))
    width = 0.8 / max(len(groups), 1)
    plt.figure(figsize=(8, 5))
    for i, gr in enumerate(groups):
        ys = []
        for x in xs:
            hit = d[(d[x_col].eq(x)) & (d[group_col].eq(gr))]
            ys.append(hit[y_col].iloc[0] if not hit.empty else np.nan)
        plt.bar(xpos + (i - (len(groups)-1)/2) * width, ys, width=width, label=str(gr))
    plt.xticks(xpos, [str(x) for x in xs])
    plt.xlabel(xlabel or x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def write_basic_outputs(input_dir, benchmark, out_dir, logs_dir, clean):
    ensure_dir(out_dir, clean)
    tables = out_dir / "tables"
    figs = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    rounds, final, status = collect(input_dir, benchmark)
    final, gpu_proc, gpu_run = add_gpu_and_efficiency(final, logs_dir)

    rounds.to_csv(tables / f"{benchmark}_round_metrics_raw.csv", index=False)
    final.to_csv(tables / f"{benchmark}_final_metrics_raw.csv", index=False)
    status.to_csv(tables / f"{benchmark}_run_status.csv", index=False)
    if not gpu_proc.empty:
        gpu_proc.to_csv(tables / f"{benchmark}_gpu_by_process.csv", index=False)
    if not gpu_run.empty:
        gpu_run.to_csv(tables / f"{benchmark}_gpu_by_run.csv", index=False)
    return rounds, final, status, tables, figs


def analyze_main(args):
    out = Path(args.out) / "main_benchmark"
    rounds, final, status, tables, figs = write_basic_outputs(
        Path(args.main_input), "main_benchmark", out, Path(args.logs) if args.logs else None, args.clean
    )
    if final.empty:
        print("No main benchmark metrics found.")
        return

    final = final[
        final["protocol"].isin(["splitfedv1", "splitfedv2"])
        & final["num_clients"].eq(4)
        & final["alpha"].isin([100.0, 1.0, 0.5])
        & final["split_point"].isin(["block1", "block2", "block3"])
        & final["rounds"].eq(50)
    ].copy()
    rounds = rounds[rounds["run_name"].isin(set(final["run_name"]))].copy()

    final.to_csv(tables / "main_clean_final_metrics.csv", index=False)
    rounds.to_csv(tables / "main_clean_round_metrics.csv", index=False)

    metrics = existing(final, DETECTION_COLS + SYSTEM_COLS + [
        "auprc_per_second", "auprc_per_comm_mb",
        "server_gpu_util_mean_pct", "server_gpu_memory_max_mb",
        "client_gpu_util_mean_pct", "client_gpu_memory_max_mb",
    ])

    summary = mean_std_table(final, ["protocol", "alpha", "split_point"], metrics)
    summary.to_csv(tables / "main_summary_by_protocol_alpha_split.csv", index=False)
    paper_table(summary, ["protocol", "alpha", "split_point"], [
        "eval_auprc", "eval_f1", "eval_recall", "eval_precision", "eval_auroc",
        "eval_bmr_risk", "fit_elapsed_time", "fit_cumulative_activation_comm_mb",
        "server_gpu_memory_max_mb", "client_gpu_memory_max_mb"
    ]).to_csv(tables / "paper_table_main_summary_mean_std.csv", index=False)

    final[existing(final, META_COLS + DETECTION_COLS + SYSTEM_COLS + [
        "auprc_per_second", "auprc_per_comm_mb", "server_gpu_util_mean_pct",
        "client_gpu_util_mean_pct", "overall_gpu_memory_max_mb"
    ])].sort_values(["alpha", "seed", "split_point", "protocol"]).to_csv(
        tables / "paper_table_main_final_by_run.csv", index=False
    )

    # Protocol deltas.
    delta_metrics = existing(final, ["eval_auprc", "eval_f1", "eval_recall", "eval_bmr_risk", "fit_elapsed_time", "fit_cumulative_activation_comm_mb"])
    piv = final.pivot_table(index=["num_clients","alpha","seed","rounds","split_point"], columns="protocol", values=delta_metrics, aggfunc="mean")
    delta = None
    for m in delta_metrics:
        if (m, "splitfedv1") in piv.columns and (m, "splitfedv2") in piv.columns:
            part = (piv[(m, "splitfedv2")] - piv[(m, "splitfedv1")]).reset_index(name=f"delta_{m}_splitfedv2_minus_splitfedv1")
            delta = part if delta is None else delta.merge(part, on=["num_clients","alpha","seed","rounds","split_point"], how="outer")
    if delta is not None:
        delta.to_csv(tables / "paper_table_protocol_deltas_splitfedv2_minus_splitfedv1.csv", index=False)

    # Trade-off ranking.
    rank_cols = existing(final, META_COLS + ["eval_auprc", "eval_f1", "eval_bmr_risk", "fit_elapsed_time", "fit_cumulative_activation_comm_mb", "auprc_per_second", "auprc_per_comm_mb"])
    ranking = final[rank_cols].sort_values("eval_auprc", ascending=False) if "eval_auprc" in final.columns else final[rank_cols]
    ranking.to_csv(tables / "paper_table_tradeoff_ranking.csv", index=False)

    # Figures.
    plot_rounds(
        rounds[(rounds["alpha"].eq(0.5)) & (rounds["seed"].eq(10)) & (rounds["split_point"].eq("block1"))],
        "protocol", "eval_auprc",
        "AUPRC over rounds by protocol\nclients=4, alpha=0.5, seed=10, split=block1",
        "AUPRC", figs / "fig_main_protocol_auprc_rounds_alpha0.5_seed10_block1.png"
    )
    for p in ["splitfedv1", "splitfedv2"]:
        plot_rounds(
            rounds[(rounds["protocol"].eq(p)) & (rounds["alpha"].eq(0.5)) & (rounds["seed"].eq(10))],
            "split_point", "eval_auprc",
            f"AUPRC over rounds by split point\nprotocol={p}, clients=4, alpha=0.5, seed=10",
            "AUPRC", figs / f"fig_main_splitpoint_auprc_rounds_{p}_alpha0.5_seed10.png"
        )

    block1 = final[final["split_point"].eq("block1")].copy()
    if not block1.empty:
        block1["alpha_label"] = block1["alpha"].map(alpha_label)
        a_order = [alpha_label(x) for x in alpha_order(block1["alpha"])]
        plot_category_lines(block1, "alpha_label", "protocol", "eval_auprc",
                            "Final AUPRC by protocol and non-IID level\nclients=4, split=block1, mean over seeds",
                            "Final AUPRC", figs / "fig_main_final_auprc_by_alpha_protocol_block1.png",
                            xlabel="Dirichlet alpha", order=a_order)
        plot_category_lines(block1, "alpha_label", "protocol", "eval_f1",
                            "Final F1-score by protocol and non-IID level\nclients=4, split=block1, mean over seeds",
                            "Final F1-score", figs / "fig_main_final_f1_by_alpha_protocol_block1.png",
                            xlabel="Dirichlet alpha", order=a_order)
        plot_category_lines(block1, "alpha_label", "protocol", "eval_bmr_risk",
                            "Final BMR risk by protocol and non-IID level\nclients=4, split=block1, mean over seeds",
                            "BMR risk", figs / "fig_main_final_bmr_risk_by_alpha_protocol_block1.png",
                            xlabel="Dirichlet alpha", order=a_order)

    alpha05 = final[final["alpha"].eq(0.5)]
    plot_grouped_bars(alpha05, "split_point", "protocol", "eval_auprc",
                      "Final AUPRC by split point\nclients=4, alpha=0.5, mean over seeds",
                      "Final AUPRC", figs / "fig_main_final_auprc_by_split_protocol_alpha0.5.png",
                      xlabel="Split point", order=["block1","block2","block3"])
    plot_grouped_bars(alpha05, "split_point", "protocol", "fit_elapsed_time",
                      "Elapsed time by split point\nclients=4, alpha=0.5, mean over seeds",
                      "Elapsed time (s)", figs / "fig_main_elapsed_time_by_split_protocol_alpha0.5.png",
                      xlabel="Split point", order=["block1","block2","block3"])
    if "fit_cumulative_activation_comm_mb" in final.columns:
        plot_grouped_bars(alpha05, "split_point", "protocol", "fit_cumulative_activation_comm_mb",
                          "Estimated communication by split point\nclients=4, alpha=0.5, mean over seeds",
                          "Communication (MB)", figs / "fig_main_comm_by_split_protocol_alpha0.5.png",
                          xlabel="Split point", order=["block1","block2","block3"])

    if {"fit_cumulative_activation_comm_mb", "eval_auprc", "protocol"}.issubset(final.columns):
        plt.figure(figsize=(8,5))
        for p, g in final.groupby("protocol"):
            plt.scatter(g["fit_cumulative_activation_comm_mb"], g["eval_auprc"], label=p)
        plt.xlabel("Estimated cumulative activation communication (MB)")
        plt.ylabel("Final AUPRC")
        plt.title("Final AUPRC vs estimated communication\nmain benchmark")
        plt.legend(); plt.tight_layout()
        plt.savefig(figs / "fig_main_tradeoff_auprc_vs_comm.png", dpi=300); plt.close()

    (out / "README.txt").write_text("Main benchmark paper tables and figures.\n", encoding="utf-8")
    print(f"Main benchmark analysis written to {out}")


def analyze_clients(args):
    out = Path(args.out) / "client_count_ablation"
    rounds, final, status, tables, figs = write_basic_outputs(
        Path(args.clients_input), "client_count_ablation", out, Path(args.logs) if args.logs else None, args.clean
    )

    # Optionally add c4 baseline from main benchmark when client folder only has c2/c3.
    if args.add_main_c4:
        mr, mf, _ = collect(Path(args.main_input), "client_count_c4_from_main")
        mf, _, _ = add_gpu_and_efficiency(mf, Path(args.logs) if args.logs else None)
        mf = mf[(mf["num_clients"].eq(4)) & (mf["alpha"].eq(0.5)) & (mf["split_point"].eq("block1")) & (mf["rounds"].eq(50))]
        mr = mr[mr["run_name"].isin(set(mf["run_name"]))]
        if not mf.empty and (final.empty or 4 not in set(final["num_clients"])):
            final = pd.concat([final, mf], ignore_index=True)
            rounds = pd.concat([rounds, mr], ignore_index=True)

    if final.empty:
        print("No client-count metrics found.")
        return

    final = final[
        final["protocol"].isin(["splitfedv1","splitfedv2"])
        & final["num_clients"].isin([2,3,4])
        & final["alpha"].eq(0.5)
        & final["split_point"].eq("block1")
        & final["rounds"].eq(50)
    ].copy()
    rounds = rounds[rounds["run_name"].isin(set(final["run_name"]))].copy()

    final.to_csv(tables / "client_clean_final_metrics.csv", index=False)
    rounds.to_csv(tables / "client_clean_round_metrics.csv", index=False)

    metrics = existing(final, DETECTION_COLS + SYSTEM_COLS + ["auprc_per_second", "auprc_per_comm_mb", "server_gpu_memory_max_mb", "client_gpu_memory_max_mb"])
    summary = mean_std_table(final, ["protocol","num_clients"], metrics)
    summary.to_csv(tables / "client_summary_by_protocol_num_clients.csv", index=False)
    paper_table(summary, ["protocol","num_clients"], [
        "eval_auprc", "eval_f1", "eval_recall", "eval_precision", "eval_bmr_risk",
        "fit_elapsed_time", "fit_cumulative_activation_comm_mb", "server_gpu_memory_max_mb", "client_gpu_memory_max_mb"
    ]).to_csv(tables / "paper_table_client_count_summary_mean_std.csv", index=False)

    final[existing(final, META_COLS + DETECTION_COLS + SYSTEM_COLS + ["auprc_per_second","auprc_per_comm_mb"])]\
        .sort_values(["protocol","seed","num_clients"]).to_csv(tables / "paper_table_client_count_final_by_run.csv", index=False)

    order = [2,3,4]
    plot_category_lines(final, "num_clients", "protocol", "eval_auprc",
                        "Final AUPRC by number of clients\nalpha=0.5, split=block1, mean over seeds",
                        "Final AUPRC", figs / "fig_client_final_auprc_by_num_clients.png",
                        xlabel="Number of clients", order=order)
    plot_category_lines(final, "num_clients", "protocol", "eval_f1",
                        "Final F1-score by number of clients\nalpha=0.5, split=block1, mean over seeds",
                        "Final F1-score", figs / "fig_client_final_f1_by_num_clients.png",
                        xlabel="Number of clients", order=order)
    plot_category_lines(final, "num_clients", "protocol", "fit_elapsed_time",
                        "Elapsed time by number of clients\nalpha=0.5, split=block1, mean over seeds",
                        "Elapsed time (s)", figs / "fig_client_elapsed_time_by_num_clients.png",
                        xlabel="Number of clients", order=order)
    if "fit_cumulative_activation_comm_mb" in final.columns:
        plot_category_lines(final, "num_clients", "protocol", "fit_cumulative_activation_comm_mb",
                            "Estimated communication by number of clients\nalpha=0.5, split=block1, mean over seeds",
                            "Communication (MB)", figs / "fig_client_comm_by_num_clients.png",
                            xlabel="Number of clients", order=order)

    for p in ["splitfedv1", "splitfedv2"]:
        plot_rounds(
            rounds[(rounds["protocol"].eq(p)) & (rounds["seed"].eq(10))],
            "num_clients", "eval_auprc",
            f"AUPRC over rounds by number of clients\nprotocol={p}, alpha=0.5, split=block1, seed=10",
            "AUPRC", figs / f"fig_client_auprc_rounds_by_clients_{p}_seed10.png"
        )

    if {"fit_elapsed_time", "eval_auprc", "protocol"}.issubset(final.columns):
        plt.figure(figsize=(8,5))
        for p, g in final.groupby("protocol"):
            plt.scatter(g["fit_elapsed_time"], g["eval_auprc"], label=p)
        plt.xlabel("Elapsed time (s)")
        plt.ylabel("Final AUPRC")
        plt.title("Final AUPRC vs elapsed time\nclient-count ablation")
        plt.legend(); plt.tight_layout()
        plt.savefig(figs / "fig_client_tradeoff_auprc_vs_elapsed_time.png", dpi=300); plt.close()

    (out / "README.txt").write_text("Client-count ablation paper tables and figures.\n", encoding="utf-8")
    print(f"Client-count ablation analysis written to {out}")


def analyze_stress(args):
    out = Path(args.out) / "stress_non_iid"
    rounds, final, status, tables, figs = write_basic_outputs(
        Path(args.stress_input), "stress_non_iid", out, Path(args.logs) if args.logs else None, args.clean
    )

    if not status.empty:
        status["completed_like"] = status["status"].eq("METRICS_FOUND")
        feasible = status.groupby(["protocol","num_clients"], dropna=False).agg(
            n_runs=("run_name","count"),
            n_metrics_found=("completed_like","sum"),
        ).reset_index()
        feasible["completion_rate"] = feasible["n_metrics_found"] / feasible["n_runs"]
        feasible.to_csv(tables / "paper_table_stress_feasibility_by_protocol_clients.csv", index=False)

        if not feasible.empty:
            # Completion rate plot.
            protocols = list(feasible["protocol"].dropna().unique())
            clients = sorted(feasible["num_clients"].dropna().unique())
            x = np.arange(len(clients)); width = 0.8 / max(len(protocols),1)
            plt.figure(figsize=(8,5))
            for i,p in enumerate(protocols):
                g = feasible[feasible["protocol"].eq(p)]
                ys = []
                for c in clients:
                    h = g[g["num_clients"].eq(c)]
                    ys.append(h["completion_rate"].iloc[0] if not h.empty else 0)
                plt.bar(x + (i - (len(protocols)-1)/2)*width, ys, width=width, label=str(p))
            plt.xticks(x, [str(c) for c in clients])
            plt.ylim(0,1.05)
            plt.xlabel("Number of clients")
            plt.ylabel("Completion rate")
            plt.title("Severe non-IID feasibility\nalpha=0.1, split=block1")
            plt.legend(); plt.tight_layout()
            plt.savefig(figs / "fig_stress_completion_rate_by_clients_protocol.png", dpi=300); plt.close()

    if final.empty:
        print("No completed stress-test metrics found; feasibility table still written.")
        return

    final = final[(final["alpha"].eq(0.1)) & (final["split_point"].eq("block1"))].copy()
    final.to_csv(tables / "stress_clean_final_metrics.csv", index=False)
    summary = mean_std_table(final, ["protocol","num_clients"], existing(final, DETECTION_COLS + SYSTEM_COLS))
    summary.to_csv(tables / "stress_summary_by_protocol_clients.csv", index=False)
    paper_table(summary, ["protocol","num_clients"], ["eval_auprc","eval_f1","eval_recall","eval_bmr_risk","fit_elapsed_time"])\
        .to_csv(tables / "paper_table_stress_completed_metrics_mean_std.csv", index=False)

    final[existing(final, META_COLS + DETECTION_COLS + SYSTEM_COLS)].sort_values(["protocol","num_clients","seed"])\
        .to_csv(tables / "paper_table_stress_final_by_run.csv", index=False)

    plot_category_lines(final, "num_clients", "protocol", "eval_auprc",
                        "Final AUPRC in severe non-IID stress test\nalpha=0.1, split=block1",
                        "Final AUPRC", figs / "fig_stress_final_auprc_by_clients_protocol.png",
                        xlabel="Number of clients", order=[2,3,4])
    plot_category_lines(final, "num_clients", "protocol", "eval_bmr_risk",
                        "Final BMR risk in severe non-IID stress test\nalpha=0.1, split=block1",
                        "BMR risk", figs / "fig_stress_final_bmr_risk_by_clients_protocol.png",
                        xlabel="Number of clients", order=[2,3,4])
    print(f"Stress-test analysis written to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["main", "clients", "stress", "all"], default="all")
    ap.add_argument("--main-input", default="outputs/main_benchmark")
    ap.add_argument("--clients-input", default="outputs/client_count_ablation")
    ap.add_argument("--stress-input", default="outputs/stress_non_iid")
    ap.add_argument("--logs", default="logs/creditcard")
    ap.add_argument("--out", default="paper_results")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--add-main-c4", action="store_true", help="For client ablation, add missing c4 baseline from main benchmark.")
    args = ap.parse_args()

    if args.benchmark in ["main", "all"]:
        analyze_main(args)
    if args.benchmark in ["clients", "all"]:
        analyze_clients(args)
    if args.benchmark in ["stress", "all"]:
        analyze_stress(args)


if __name__ == "__main__":
    main()
