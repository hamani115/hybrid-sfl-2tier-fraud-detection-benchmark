from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    summary_dir = Path("results/summary")
    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    round_df = pd.read_csv(summary_dir / "round_metrics.csv")
    final_df = pd.read_csv(summary_dir / "final_metrics.csv")

    # Plot 1: AUPRC over rounds
    if "eval_auprc" in round_df.columns:
        plt.figure()
        for run_name, group in round_df.groupby("run_name"):
            plt.plot(group["round"], group["eval_auprc"], label=run_name)

        plt.xlabel("Round")
        plt.ylabel("AUPRC")
        plt.title("AUPRC over training rounds")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(plots_dir / "auprc_over_rounds.png", dpi=300)
        plt.close()

    # Plot 2: F1 over rounds
    if "eval_f1" in round_df.columns:
        plt.figure()
        for run_name, group in round_df.groupby("run_name"):
            plt.plot(group["round"], group["eval_f1"], label=run_name)

        plt.xlabel("Round")
        plt.ylabel("F1-score")
        plt.title("F1-score over training rounds")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(plots_dir / "f1_over_rounds.png", dpi=300)
        plt.close()

    # Plot 3: Final AUPRC by protocol and alpha
    if {"protocol", "alpha", "eval_auprc"}.issubset(final_df.columns):
        grouped = (
            final_df.groupby(["protocol", "alpha"])["eval_auprc"]
            .mean()
            .reset_index()
        )

        plt.figure()
        for protocol, group in grouped.groupby("protocol"):
            plt.plot(group["alpha"], group["eval_auprc"], marker="o", label=protocol)

        plt.xlabel("Dirichlet alpha")
        plt.ylabel("Final AUPRC")
        plt.title("Final AUPRC by protocol and non-IID level")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "final_auprc_by_alpha.png", dpi=300)
        plt.close()

    # Plot 4: Quality vs time
    if {"eval_elapsed_time", "eval_auprc"}.issubset(round_df.columns):
        plt.figure()
        for run_name, group in round_df.groupby("run_name"):
            plt.plot(group["eval_elapsed_time"], group["eval_auprc"], label=run_name)

        plt.xlabel("Elapsed time (s)")
        plt.ylabel("AUPRC")
        plt.title("AUPRC vs elapsed time")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(plots_dir / "auprc_vs_elapsed_time.png", dpi=300)
        plt.close()

    print(f"Plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
