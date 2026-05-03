from pathlib import Path

import pandas as pd


AUPRC_TARGETS = [0.75, 0.80, 0.85]
F1_TARGETS = [0.75, 0.80]


def first_reach(group, metric_col, target, value_col):
    if metric_col not in group.columns or value_col not in group.columns:
        return None

    reached = group[group[metric_col] >= target]
    if reached.empty:
        return None

    return reached.iloc[0][value_col]


def main():
    summary_dir = Path("results/summary")
    out_path = summary_dir / "tradeoff_metrics.csv"

    round_df = pd.read_csv(summary_dir / "round_metrics.csv")
    rows = []

    for run_name, group in round_df.groupby("run_name"):
        group = group.sort_values("round").copy()

        if "fit_activation_comm_mb" in group.columns:
            group["cumulative_activation_comm_mb"] = group[
                "fit_activation_comm_mb"
            ].fillna(0).cumsum()
        else:
            group["cumulative_activation_comm_mb"] = None

        final = group.iloc[-1].to_dict()

        row = {
            "run_name": run_name,
            "protocol": final.get("protocol"),
            "num_clients": final.get("num_clients"),
            "alpha": final.get("alpha"),
            "seed": final.get("seed"),
            "rounds": final.get("rounds"),
            "split_point": final.get("split_point"),
            "run_id": final.get("run_id"),
            "final_round": final.get("round"),
            "final_auprc": final.get("eval_auprc"),
            "final_auroc": final.get("eval_auroc"),
            "final_f1": final.get("eval_f1"),
            "final_precision": final.get("eval_precision"),
            "final_recall": final.get("eval_recall"),
            "final_bmr_risk": final.get("eval_bmr_risk"),
            "final_bmr_f1": final.get("eval_bmr_f1"),
            "final_elapsed_time": final.get("fit_elapsed_time"),
            "final_round_time": final.get("fit_round_time"),
            "final_activation_comm_mb": final.get("cumulative_activation_comm_mb"),
        }

        # Quality per time
        if row["final_elapsed_time"] and row["final_elapsed_time"] > 0:
            row["auprc_per_second"] = row["final_auprc"] / row["final_elapsed_time"]
            row["f1_per_second"] = row["final_f1"] / row["final_elapsed_time"]
        else:
            row["auprc_per_second"] = None
            row["f1_per_second"] = None

        # Quality per communication
        if row["final_activation_comm_mb"] and row["final_activation_comm_mb"] > 0:
            row["auprc_per_comm_mb"] = row["final_auprc"] / row["final_activation_comm_mb"]
            row["f1_per_comm_mb"] = row["final_f1"] / row["final_activation_comm_mb"]
        else:
            row["auprc_per_comm_mb"] = None
            row["f1_per_comm_mb"] = None

        for target in AUPRC_TARGETS:
            row[f"round_to_auprc_{target}"] = first_reach(
                group, "eval_auprc", target, "round"
            )
            row[f"time_to_auprc_{target}"] = first_reach(
                group, "eval_auprc", target, "fit_elapsed_time"
            )
            row[f"comm_to_auprc_{target}_mb"] = first_reach(
                group, "eval_auprc", target, "cumulative_activation_comm_mb"
            )

        for target in F1_TARGETS:
            row[f"round_to_f1_{target}"] = first_reach(
                group, "eval_f1", target, "round"
            )
            row[f"time_to_f1_{target}"] = first_reach(
                group, "eval_f1", target, "fit_elapsed_time"
            )

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)

    print(f"Wrote {out_path} with {len(out_df)} rows")
    print(out_df.tail())


if __name__ == "__main__":
    main()
