from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


def safe_filename(name: str) -> str:
    """Make run/setup names safe for folder/file names."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def get_time_column(df: pd.DataFrame):
    """
    Depending on how metrics were stored, elapsed time may come from fit or eval.
    Prefer fit_elapsed_time because communication metrics are also fit-side.
    """
    if "fit_elapsed_time" in df.columns:
        return "fit_elapsed_time"
    if "eval_elapsed_time" in df.columns:
        return "eval_elapsed_time"
    return None


def get_round_time_column(df: pd.DataFrame):
    if "fit_round_time" in df.columns:
        return "fit_round_time"
    if "eval_round_time" in df.columns:
        return "eval_round_time"
    return None


def get_setup_columns(df: pd.DataFrame):
    """
    Columns that define the experimental setup, excluding protocol.

    We intentionally do NOT include:
      - protocol: because this is what we compare
      - run_name: unique run identifier
      - run_id: Slurm/job/run identifier
      - round: time axis
    """
    possible_setup_cols = [
        "num_clients",
        "alpha",
        "seed",
        "rounds",
        # Future optional columns if you later add them to collect_results.py
        "split_point",
        "batch_size",
        "local_epochs",
        "class_weight",
        "model",
        "dataset",
    ]

    return [col for col in possible_setup_cols if col in df.columns]


def setup_to_string(setup_values: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in setup_values.items())


def plot_line(
    group: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
):
    if x_col not in group.columns or y_col not in group.columns:
        return

    data = group[[x_col, y_col]].dropna().sort_values(x_col)

    if data.empty:
        return

    plt.figure()
    plt.plot(data[x_col], data[y_col], marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_protocol_comparison(
    df: pd.DataFrame,
    setup_cols: list[str],
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    comparison_dir: Path,
    title_metric: str,
):
    """
    Make one combined plot per setup group.

    Example:
      Same num_clients, alpha, seed, rounds
      but different protocols.

    Each protocol gets one line.
    If there are repeated runs for the same protocol/setup/round,
    the script averages them.
    """
    required_cols = set(setup_cols + ["protocol", x_col, y_col])
    if not required_cols.issubset(df.columns):
        return

    for setup_key, setup_group in df.groupby(setup_cols, dropna=False):
        if not isinstance(setup_key, tuple):
            setup_key = (setup_key,)

        setup_values = dict(zip(setup_cols, setup_key))
        protocols = sorted(setup_group["protocol"].dropna().unique())

        # Only make combined protocol plots if at least two protocols exist
        # for the same setup.
        if len(protocols) < 2:
            continue

        setup_text = setup_to_string(setup_values)
        setup_safe = safe_filename(setup_text)

        setup_plot_dir = comparison_dir / setup_safe
        setup_plot_dir.mkdir(parents=True, exist_ok=True)

        plt.figure()

        for protocol, protocol_group in setup_group.groupby("protocol"):
            data = protocol_group[[x_col, y_col, "round"]].dropna().copy()

            if data.empty:
                continue

            # Average repeated runs for the same protocol/setup/round.
            data = (
                data.groupby("round")[[x_col, y_col]]
                .mean()
                .reset_index()
                .sort_values(x_col)
            )

            plt.plot(
                data[x_col],
                data[y_col],
                marker="o",
                label=f"{protocol}",
            )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{title_metric}\n{setup_text}")
        plt.legend(title="Protocol")
        plt.tight_layout()
        plt.savefig(setup_plot_dir / filename, dpi=300)
        plt.close()


def plot_protocol_comparison_comm(
    df: pd.DataFrame,
    setup_cols: list[str],
    comparison_dir: Path,
):
    """
    Plot AUPRC vs cumulative estimated activation communication.

    Each protocol gets one line for the same setup.
    """
    required_cols = set(setup_cols + ["protocol", "round", "fit_activation_comm_mb", "eval_auprc"])
    if not required_cols.issubset(df.columns):
        return

    for setup_key, setup_group in df.groupby(setup_cols, dropna=False):
        if not isinstance(setup_key, tuple):
            setup_key = (setup_key,)

        setup_values = dict(zip(setup_cols, setup_key))
        protocols = sorted(setup_group["protocol"].dropna().unique())

        if len(protocols) < 2:
            continue

        setup_text = setup_to_string(setup_values)
        setup_safe = safe_filename(setup_text)

        setup_plot_dir = comparison_dir / setup_safe
        setup_plot_dir.mkdir(parents=True, exist_ok=True)

        plt.figure()

        for protocol, protocol_group in setup_group.groupby("protocol"):
            data = protocol_group[
                ["run_name", "round", "fit_activation_comm_mb", "eval_auprc"]
            ].dropna().copy()

            if data.empty:
                continue

            # First compute cumulative communication per run.
            run_rows = []
            for run_name, run_group in data.groupby("run_name"):
                run_group = run_group.sort_values("round").copy()
                run_group["cumulative_activation_comm_mb"] = run_group[
                    "fit_activation_comm_mb"
                ].cumsum()
                run_rows.append(run_group)

            data = pd.concat(run_rows, ignore_index=True)

            # Then average repeated runs for the same protocol/setup/round.
            data = (
                data.groupby("round")[["cumulative_activation_comm_mb", "eval_auprc"]]
                .mean()
                .reset_index()
                .sort_values("cumulative_activation_comm_mb")
            )

            plt.plot(
                data["cumulative_activation_comm_mb"],
                data["eval_auprc"],
                marker="o",
                label=f"{protocol}",
            )

        plt.xlabel("Estimated cumulative activation communication (MB)")
        plt.ylabel("AUPRC")
        plt.title(f"AUPRC vs estimated communication cost\n{setup_text}")
        plt.legend(title="Protocol")
        plt.tight_layout()
        plt.savefig(setup_plot_dir / "auprc_vs_estimated_communication_mb.png", dpi=300)
        plt.close()


def main():
    summary_dir = Path("results/summary")
    plots_dir = Path("results/plots")

    per_run_dir = plots_dir / "per_run"
    comparison_dir = plots_dir / "protocol_comparison"
    summary_plots_dir = plots_dir / "summary"

    per_run_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    summary_plots_dir.mkdir(parents=True, exist_ok=True)

    round_df = pd.read_csv(summary_dir / "round_metrics.csv")
    final_df = pd.read_csv(summary_dir / "final_metrics.csv")

    time_col = get_time_column(round_df)
    round_time_col = get_round_time_column(round_df)
    setup_cols = get_setup_columns(round_df)

    print(f"Using setup columns for protocol comparison: {setup_cols}")

    # ------------------------------------------------------------------
    # Per-run plots
    # ------------------------------------------------------------------
    for run_name, group in round_df.groupby("run_name"):
        group = group.sort_values("round").copy()

        run_safe = safe_filename(run_name)
        run_plot_dir = per_run_dir / run_safe
        run_plot_dir.mkdir(parents=True, exist_ok=True)

        plot_line(
            group=group,
            x_col="round",
            y_col="eval_auprc",
            title=f"AUPRC over rounds\n{run_name}",
            xlabel="Round",
            ylabel="AUPRC",
            out_path=run_plot_dir / "auprc_over_rounds.png",
        )

        plot_line(
            group=group,
            x_col="round",
            y_col="eval_f1",
            title=f"F1-score over rounds\n{run_name}",
            xlabel="Round",
            ylabel="F1-score",
            out_path=run_plot_dir / "f1_over_rounds.png",
        )

        plot_line(
            group=group,
            x_col="round",
            y_col="eval_precision",
            title=f"Precision over rounds\n{run_name}",
            xlabel="Round",
            ylabel="Precision",
            out_path=run_plot_dir / "precision_over_rounds.png",
        )

        plot_line(
            group=group,
            x_col="round",
            y_col="eval_recall",
            title=f"Recall over rounds\n{run_name}",
            xlabel="Round",
            ylabel="Recall",
            out_path=run_plot_dir / "recall_over_rounds.png",
        )

        plot_line(
            group=group,
            x_col="round",
            y_col="eval_auroc",
            title=f"AUROC over rounds\n{run_name}",
            xlabel="Round",
            ylabel="AUROC",
            out_path=run_plot_dir / "auroc_over_rounds.png",
        )

        plot_line(
            group=group,
            x_col="round",
            y_col="eval_bmr_risk",
            title=f"BMR risk over rounds\n{run_name}",
            xlabel="Round",
            ylabel="BMR risk",
            out_path=run_plot_dir / "bmr_risk_over_rounds.png",
        )

        if time_col is not None:
            plot_line(
                group=group,
                x_col=time_col,
                y_col="eval_auprc",
                title=f"AUPRC vs elapsed time\n{run_name}",
                xlabel="Elapsed time (s)",
                ylabel="AUPRC",
                out_path=run_plot_dir / "auprc_vs_elapsed_time.png",
            )

        if round_time_col is not None:
            plot_line(
                group=group,
                x_col="round",
                y_col=round_time_col,
                title=f"Round time over rounds\n{run_name}",
                xlabel="Round",
                ylabel="Round time (s)",
                out_path=run_plot_dir / "round_time_over_rounds.png",
            )

        if {"fit_activation_comm_mb", "eval_auprc"}.issubset(group.columns):
            comm_group = group[
                ["round", "fit_activation_comm_mb", "eval_auprc"]
            ].dropna().sort_values("round").copy()

            if not comm_group.empty:
                comm_group["cumulative_activation_comm_mb"] = comm_group[
                    "fit_activation_comm_mb"
                ].cumsum()

                plt.figure()
                plt.plot(
                    comm_group["cumulative_activation_comm_mb"],
                    comm_group["eval_auprc"],
                    marker="o",
                )
                plt.xlabel("Estimated cumulative activation communication (MB)")
                plt.ylabel("AUPRC")
                plt.title(f"AUPRC vs estimated communication cost\n{run_name}")
                plt.tight_layout()
                plt.savefig(run_plot_dir / "auprc_vs_estimated_communication_mb.png", dpi=300)
                plt.close()

        group.to_csv(run_plot_dir / "round_metrics_for_this_run.csv", index=False)

    # ------------------------------------------------------------------
    # Protocol-comparison plots
    # Same setup values, different protocol lines.
    # ------------------------------------------------------------------
    if setup_cols:
        plot_protocol_comparison(
            df=round_df,
            setup_cols=setup_cols,
            x_col="round",
            y_col="eval_auprc",
            xlabel="Round",
            ylabel="AUPRC",
            filename="auprc_over_rounds.png",
            comparison_dir=comparison_dir,
            title_metric="AUPRC over rounds by protocol",
        )

        plot_protocol_comparison(
            df=round_df,
            setup_cols=setup_cols,
            x_col="round",
            y_col="eval_f1",
            xlabel="Round",
            ylabel="F1-score",
            filename="f1_over_rounds.png",
            comparison_dir=comparison_dir,
            title_metric="F1-score over rounds by protocol",
        )

        plot_protocol_comparison(
            df=round_df,
            setup_cols=setup_cols,
            x_col="round",
            y_col="eval_precision",
            xlabel="Round",
            ylabel="Precision",
            filename="precision_over_rounds.png",
            comparison_dir=comparison_dir,
            title_metric="Precision over rounds by protocol",
        )

        plot_protocol_comparison(
            df=round_df,
            setup_cols=setup_cols,
            x_col="round",
            y_col="eval_recall",
            xlabel="Round",
            ylabel="Recall",
            filename="recall_over_rounds.png",
            comparison_dir=comparison_dir,
            title_metric="Recall over rounds by protocol",
        )

        plot_protocol_comparison(
            df=round_df,
            setup_cols=setup_cols,
            x_col="round",
            y_col="eval_auroc",
            xlabel="Round",
            ylabel="AUROC",
            filename="auroc_over_rounds.png",
            comparison_dir=comparison_dir,
            title_metric="AUROC over rounds by protocol",
        )

        plot_protocol_comparison(
            df=round_df,
            setup_cols=setup_cols,
            x_col="round",
            y_col="eval_bmr_risk",
            xlabel="Round",
            ylabel="BMR risk",
            filename="bmr_risk_over_rounds.png",
            comparison_dir=comparison_dir,
            title_metric="BMR risk over rounds by protocol",
        )

        if time_col is not None:
            plot_protocol_comparison(
                df=round_df,
                setup_cols=setup_cols,
                x_col=time_col,
                y_col="eval_auprc",
                xlabel="Elapsed time (s)",
                ylabel="AUPRC",
                filename="auprc_vs_elapsed_time.png",
                comparison_dir=comparison_dir,
                title_metric="AUPRC vs elapsed time by protocol",
            )

        if round_time_col is not None:
            plot_protocol_comparison(
                df=round_df,
                setup_cols=setup_cols,
                x_col="round",
                y_col=round_time_col,
                xlabel="Round",
                ylabel="Round time (s)",
                filename="round_time_over_rounds.png",
                comparison_dir=comparison_dir,
                title_metric="Round time over rounds by protocol",
            )

        plot_protocol_comparison_comm(
            df=round_df,
            setup_cols=setup_cols,
            comparison_dir=comparison_dir,
        )

    # ------------------------------------------------------------------
    # Simple high-level summary plots
    # These average over seeds/runs if present.
    # ------------------------------------------------------------------

    if {"protocol", "alpha", "eval_auprc"}.issubset(final_df.columns):
        grouped = (
            final_df.groupby(["protocol", "alpha"])["eval_auprc"]
            .mean()
            .reset_index()
        )

        plt.figure()
        for protocol, group in grouped.groupby("protocol"):
            group = group.sort_values("alpha")
            plt.plot(group["alpha"], group["eval_auprc"], marker="o", label=protocol)

        plt.xlabel("Dirichlet alpha")
        plt.ylabel("Final AUPRC")
        plt.title("Final AUPRC by protocol and non-IID level")
        plt.legend(title="Protocol")
        plt.tight_layout()
        plt.savefig(summary_plots_dir / "final_auprc_by_alpha.png", dpi=300)
        plt.close()

    if {"protocol", "alpha", "eval_f1"}.issubset(final_df.columns):
        grouped = (
            final_df.groupby(["protocol", "alpha"])["eval_f1"]
            .mean()
            .reset_index()
        )

        plt.figure()
        for protocol, group in grouped.groupby("protocol"):
            group = group.sort_values("alpha")
            plt.plot(group["alpha"], group["eval_f1"], marker="o", label=protocol)

        plt.xlabel("Dirichlet alpha")
        plt.ylabel("Final F1-score")
        plt.title("Final F1-score by protocol and non-IID level")
        plt.legend(title="Protocol")
        plt.tight_layout()
        plt.savefig(summary_plots_dir / "final_f1_by_alpha.png", dpi=300)
        plt.close()

    print(f"Per-run plots saved to: {per_run_dir}")
    print(f"Protocol comparison plots saved to: {comparison_dir}")
    print(f"Summary plots saved to: {summary_plots_dir}")


if __name__ == "__main__":
    main()