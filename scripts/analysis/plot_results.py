from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


def safe_filename(name: str) -> str:
    """Make run names safe for folder/file names."""
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


def plot_line(group: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, out_path: Path):
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


def main():
    summary_dir = Path("results/summary")
    plots_dir = Path("results/plots")
    per_run_dir = plots_dir / "per_run"
    summary_plots_dir = plots_dir / "summary"

    per_run_dir.mkdir(parents=True, exist_ok=True)
    summary_plots_dir.mkdir(parents=True, exist_ok=True)

    round_df = pd.read_csv(summary_dir / "round_metrics.csv")
    final_df = pd.read_csv(summary_dir / "final_metrics.csv")

    time_col = get_time_column(round_df)

    # ------------------------------------------------------------------
    # Per-run plots
    # ------------------------------------------------------------------
    for run_name, group in round_df.groupby("run_name"):
        group = group.sort_values("round").copy()

        run_safe = safe_filename(run_name)
        run_plot_dir = per_run_dir / run_safe
        run_plot_dir.mkdir(parents=True, exist_ok=True)

        # 1. AUPRC over rounds
        plot_line(
            group=group,
            x_col="round",
            y_col="eval_auprc",
            title=f"AUPRC over rounds\n{run_name}",
            xlabel="Round",
            ylabel="AUPRC",
            out_path=run_plot_dir / "auprc_over_rounds.png",
        )

        # 2. F1 over rounds
        plot_line(
            group=group,
            x_col="round",
            y_col="eval_f1",
            title=f"F1-score over rounds\n{run_name}",
            xlabel="Round",
            ylabel="F1-score",
            out_path=run_plot_dir / "f1_over_rounds.png",
        )

        # 3. Precision over rounds
        plot_line(
            group=group,
            x_col="round",
            y_col="eval_precision",
            title=f"Precision over rounds\n{run_name}",
            xlabel="Round",
            ylabel="Precision",
            out_path=run_plot_dir / "precision_over_rounds.png",
        )

        # 4. Recall over rounds
        plot_line(
            group=group,
            x_col="round",
            y_col="eval_recall",
            title=f"Recall over rounds\n{run_name}",
            xlabel="Round",
            ylabel="Recall",
            out_path=run_plot_dir / "recall_over_rounds.png",
        )

        # 5. AUROC over rounds
        plot_line(
            group=group,
            x_col="round",
            y_col="eval_auroc",
            title=f"AUROC over rounds\n{run_name}",
            xlabel="Round",
            ylabel="AUROC",
            out_path=run_plot_dir / "auroc_over_rounds.png",
        )

        # 6. BMR risk over rounds
        plot_line(
            group=group,
            x_col="round",
            y_col="eval_bmr_risk",
            title=f"BMR risk over rounds\n{run_name}",
            xlabel="Round",
            ylabel="BMR risk",
            out_path=run_plot_dir / "bmr_risk_over_rounds.png",
        )

        # 7. AUPRC vs elapsed time
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

        # 8. Round time over rounds
        round_time_col = None
        if "fit_round_time" in group.columns:
            round_time_col = "fit_round_time"
        elif "eval_round_time" in group.columns:
            round_time_col = "eval_round_time"

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

        # 9. AUPRC vs estimated cumulative activation communication
        if {"fit_activation_comm_mb", "eval_auprc"}.issubset(group.columns):
            comm_group = group[["round", "fit_activation_comm_mb", "eval_auprc"]].dropna().sort_values("round").copy()

            if not comm_group.empty:
                comm_group["cumulative_activation_comm_mb"] = comm_group["fit_activation_comm_mb"].cumsum()

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

        # Save the per-run rows as CSV too, useful for checking one run quickly.
        group.to_csv(run_plot_dir / "round_metrics_for_this_run.csv", index=False)

    # ------------------------------------------------------------------
    # Summary plots across runs
    # These are still useful for comparing protocols after many runs.
    # ------------------------------------------------------------------

    # Final AUPRC by protocol and alpha
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
        plt.legend()
        plt.tight_layout()
        plt.savefig(summary_plots_dir / "final_auprc_by_alpha.png", dpi=300)
        plt.close()

    # Final F1 by protocol and alpha
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
        plt.legend()
        plt.tight_layout()
        plt.savefig(summary_plots_dir / "final_f1_by_alpha.png", dpi=300)
        plt.close()

    print(f"Per-run plots saved to: {per_run_dir}")
    print(f"Summary plots saved to: {summary_plots_dir}")


if __name__ == "__main__":
    main()